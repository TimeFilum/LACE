import os
import time
import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import visual

from lace.lace import LACE
from lace.lace_config import LACEConfig
from lace.losses import (
    accuracy_loss,
    structural_coherence_loss,
)

from config import EVALUATION_RESULTS_PATH

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


class Exp_Multiscale_LACE_Forecast(Exp_Basic):
    """
    THUML experiment wrapper for LACE.

    Responsibilities
    ----------------
    - obtain train/validation/test loaders
    - construct multi-resolution targets
    - compute train-set statistics for standardized accuracy losses
    - select the configured coherence objective
    - run optimization, validation, checkpointing, and testing

    """

    def __init__(self, args):
        # Exp_Basic calls our _build_model() during initialization.
        super().__init__(args)

        self.lace_cfg = LACEConfig.from_namespace(args)

        model = self._core_model()
        self.factors = list(model.factors)
        self.num_scales = len(self.factors)
        self.hierarchy = model.hierarchy

        if not self.factors or self.factors[0] != 1:
            raise ValueError(
                "The THUML LACE experiment expects the finest hierarchy "
                "factor to be 1."
            )

        # Accuracy-loss statistics are computed from training targets.
        self.mus = {factor: 0.0 for factor in self.factors}
        self.stds = {factor: 1.0 for factor in self.factors}

        self.patience = int(self.args.patience)
        self.best_val = float("inf")
        self.bad_epochs = 0

        self.use_wandb = bool(getattr(self.args, "use_wandb", False))
        if self.use_wandb:
            import wandb
            wandb.init(
                project="LACE",
                config=vars(args),
            )

        self._print_configuration()

    # ==============================================================
    # Model / data plumbing
    # ==============================================================

    def _core_model(self):
        """Return the underlying LACE instance with or without DataParallel."""
        return (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

    # def _build_model(self):
    #     """
    #     Build LACE directly.

    #     This deliberately bypasses the earlier exploratory implementation and
    #     keeps the experiment independent of old ms_head/multiscale_loss code.
    #     """
    #     cfg = LACEConfig.from_namespace(self.args)

    #     model = LACE(
    #         context_len=cfg.context_len,
    #         pred_len=cfg.pred_len,
    #         factors=list(cfg.factors),
    #         adapter_dim=cfg.adapter_dim,
    #         representation_norm=cfg.representation_norm,
    #         reduction=cfg.reduction,
    #         backbone_name=cfg.backbone_name,
    #     ).float()

    #     if self.args.use_multi_gpu and self.args.use_gpu:
    #         model = nn.DataParallel(
    #             model,
    #             device_ids=self.args.device_ids,
    #         )

    #     return model
    def _build_model(self):
        cfg = LACEConfig.from_namespace(self.args)

        backbone_by_model = {
            "LACE_moirai1": "Salesforce/moirai-1.0-R-small",
            "LACE_moirai2": "Salesforce/moirai-2.0-R-small",
            "LACE_timesfm2": "google/timesfm-2.0-500m-pytorch",
            "LACE_chronos2": "amazon/chronos-2",
        }

        if self.args.model not in backbone_by_model:
            raise ValueError(
                f"Unsupported LACE model variant: {self.args.model!r}"
            )

        backbone_name = backbone_by_model[self.args.model]

        model = LACE(
            context_len=cfg.context_len,
            pred_len=cfg.pred_len,
            factors=list(cfg.factors),
            adapter_dim=cfg.adapter_dim,
            representation_norm=cfg.representation_norm,
            reduction=cfg.reduction,
            backbone_name=backbone_name,
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(
                model,
                device_ids=self.args.device_ids,
            )

        return model


    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("LACE has no trainable parameters.")

        return torch.optim.Adam(
            trainable_params,
            lr=self.args.learning_rate,
        )

    def _print_configuration(self):
        model = self._core_model()

        print(
            "\n[LACE]"
            f"\n  factors          = {self.factors}"
            f"\n  target lengths   = {model.target_lengths}"
            f"\n  latent dim       = {model.latent_dim}"
            f"\n  adapter dim      = {model.adapter_dim}"
            f"\n  representation   = {self.lace_cfg.representation_norm}"
            f"\n  reduction        = {self.lace_cfg.reduction}"
            f"\n  coherence mode   = {self.lace_cfg.coherence_mode}"
            f"\n  lambda coherence = {self.lace_cfg.lambda_coherence}"
            "\n"
        )

    # ==============================================================
    # Efficiency logging
    # ==============================================================

    def _append_efficiency_record(
        self,
        setting,
        epochs_run,
        train_seconds,
        peak_allocated_mb,
        peak_reserved_mb,
    ):
        """
        Append one tab-separated efficiency record for this run.

        The file is intentionally append-only so completed runs from
        different backbones/datasets/modes accumulate in one place.
        """
        model = self._core_model()

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        trainable_pct = (
            100.0 * trainable_params / total_params
            if total_params > 0
            else float("nan")
        )

        data_name = getattr(
            self.args,
            "data_name",
            getattr(self.args, "model_id", "unknown"),
        )

        data_resolution = getattr(
            self.args,
            "data_resolution",
            "unknown",
        )

        factors_tag = "-".join(
            str(factor)
            for factor in self.factors
        )

        efficiency_file = os.path.join(
            EVALUATION_RESULTS_PATH,
            "LACE_efficiency.txt",
        )

        os.makedirs(
            EVALUATION_RESULTS_PATH,
            exist_ok=True,
        )

        columns = [
            "model",
            "dataset",
            "resolution",
            "mode",
            "lambda",
            "factors",
            "seq_len",
            "pred_len",
            "batch_size",
            "adapter_dim",
            "total_params",
            "trainable_params",
            "trainable_pct",
            "epochs_run",
            "train_seconds",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "best_val",
            "setting",
        ]

        values = [
            str(self.args.model),
            str(data_name),
            str(data_resolution),
            str(self.lace_cfg.coherence_mode),
            f"{self.lace_cfg.lambda_coherence:.8g}",
            factors_tag,
            str(self.args.seq_len),
            str(self.args.pred_len),
            str(self.args.batch_size),
            str(model.adapter_dim),
            str(total_params),
            str(trainable_params),
            f"{trainable_pct:.8f}",
            str(epochs_run),
            f"{train_seconds:.6f}",
            f"{peak_allocated_mb:.6f}",
            f"{peak_reserved_mb:.6f}",
            f"{self.best_val:.8f}",
            str(setting),
        ]

        write_header = (
            not os.path.exists(efficiency_file)
            or os.path.getsize(efficiency_file) == 0
        )

        with open(efficiency_file, "a") as f:
            if write_header:
                f.write(
                    "\t".join(columns)
                    + "\n"
                )

            f.write(
                "\t".join(values)
                + "\n"
            )

        print(
            "[EFFICIENCY] "
            f"model={self.args.model} | "
            f"dataset={data_name} | "
            f"mode={self.lace_cfg.coherence_mode} | "
            f"epochs={epochs_run} | "
            f"time={train_seconds:.2f}s | "
            f"peak_allocated={peak_allocated_mb:.2f}MB | "
            f"trainable={trainable_params:,}"
        )

        print(
            "[EFFICIENCY] appended to:",
            efficiency_file,
        )


    # ==============================================================
    # Target construction
    # ==============================================================

    def _build_targets(self, true_base):
        """
        Construct targets at all active hierarchy factors.

        Parameters
        ----------
        true_base : Tensor [B, pred_len]
            Finest-resolution future in the loader/model domain.

        Returns
        -------
        dict[int, Tensor]
            factor -> [B, L_factor]
        """
        if true_base.ndim != 2:
            raise ValueError(
                f"Expected true_base [B,T], got {true_base.shape}."
            )

        targets = {}

        for factor in self.factors:
            if factor == 1:
                targets[factor] = true_base
            else:
                targets[factor] = self.hierarchy.aggregate_prediction(
                    true_base,
                    fine_factor=1,
                    coarse_factor=factor,
                )

        return targets

    def _prepare_batch(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
    ):
        """
        Move one THUML batch to device and construct factor-keyed targets.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        true_base = (
            batch_y[:, -self.args.pred_len:, :]
            .squeeze(-1)
        )

        targets = self._build_targets(true_base)

        # Kept only for THUML call-signature compatibility.
        dec_zeros = torch.zeros_like(
            batch_y[:, -self.args.pred_len:, :]
        )
        dec_inp = torch.cat(
            [
                batch_y[:, :self.args.label_len, :],
                dec_zeros,
            ],
            dim=1,
        )

        return {
            "x_enc": batch_x,
            "x_mark_enc": batch_x_mark,
            "x_dec": dec_inp,
            "x_mark_dec": batch_y_mark,
            "y_true": targets,
        }

    # ==============================================================
    # Automatic interface sanity check
    # ==============================================================

    def _sanity_check(self, loader):
        """
        Check one real batch end-to-end before training begins.

        This replaces the old manual latent probes and target debug files.
        """
        raw_batch = next(iter(loader))
        batch = self._prepare_batch(*raw_batch)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                batch["x_enc"],
                batch["x_mark_enc"],
                batch["x_dec"],
                batch["x_mark_dec"],
            )


        expected_keys = set(self.factors)

        if set(outputs.keys()) != expected_keys:
            raise RuntimeError(
                f"Output factors {sorted(outputs.keys())} do not match "
                f"configured factors {self.factors}."
            )

        if set(batch["y_true"].keys()) != expected_keys:
            raise RuntimeError(
                f"Target factors {sorted(batch['y_true'].keys())} do not "
                f"match configured factors {self.factors}."
            )

        model = self._core_model()

        print("\n[LACE REAL-BATCH SANITY CHECK]")

        for factor, expected_len in zip(
            self.factors,
            model.target_lengths,
        ):
            pred = outputs[factor]
            true = batch["y_true"][factor]

            if pred.shape != true.shape:
                raise RuntimeError(
                    f"Shape mismatch at factor {factor}: "
                    f"prediction={pred.shape}, target={true.shape}."
                )

            if pred.shape[1] != expected_len:
                raise RuntimeError(
                    f"Unexpected output length at factor {factor}: "
                    f"got {pred.shape[1]}, expected {expected_len}."
                )

            print(
                f"  factor {factor}: "
                f"pred={tuple(pred.shape)} "
                f"target={tuple(true.shape)}"
            )

        print("  status: PASS\n")

        if was_training:
            self.model.train()

    # ==============================================================
    # Train-set statistics
    # ==============================================================

    @torch.no_grad()
    def compute_multiscale_statistics(self, train_loader):
        """
        Compute mean/std of the actual training targets at each resolution.

        Statistics are accumulated batch-by-batch in float64 and are used
        only by the standardized forecasting loss.
        """
        sums = {
            factor: torch.zeros(
                (),
                dtype=torch.float64,
                device=self.device,
            )
            for factor in self.factors
        }

        sums_sq = {
            factor: torch.zeros(
                (),
                dtype=torch.float64,
                device=self.device,
            )
            for factor in self.factors
        }

        counts = {
            factor: 0
            for factor in self.factors
        }

        for _, batch_y, _, _ in train_loader:
            batch_y = batch_y.float().to(self.device)

            true_base = (
                batch_y[:, -self.args.pred_len:, :]
                .squeeze(-1)
            )

            targets = self._build_targets(true_base)

            for factor in self.factors:
                values = targets[factor].to(torch.float64)

                sums[factor] += values.sum()
                sums_sq[factor] += (values * values).sum()
                counts[factor] += values.numel()

        for factor in self.factors:
            n = counts[factor]

            if n < 2:
                raise RuntimeError(
                    f"Not enough training values to estimate statistics "
                    f"for factor {factor}: n={n}."
                )

            mean = sums[factor] / n

            variance = (
                sums_sq[factor]
                - (sums[factor] * sums[factor]) / n
            ) / (n - 1)

            variance = torch.clamp(
                variance,
                min=1e-12,
            )

            std = torch.sqrt(variance)

            self.mus[factor] = float(mean.item())
            self.stds[factor] = float(std.item())

        print("[TRAINING TARGET STATISTICS]")
        for factor in self.factors:
            print(
                f"  factor {factor}: "
                f"mu={self.mus[factor]:.6f}, "
                f"std={self.stds[factor]:.6f}"
            )
        print()

    # ==============================================================
    # Loss routing
    # ==============================================================

    def _compute_loss(self, outputs, targets):
        """
        Compute the configured LACE objective.

        Modes
        -----
        none:
            accuracy only

        structural:
            accuracy + LACE structural regularization
        """
        weights = {
            factor: self.lace_cfg.weight_for(factor)
            for factor in self.factors
        }

        loss_acc, acc_details = accuracy_loss(
            outputs=outputs,
            targets=targets,
            factors=self.factors,
            mus=self.mus,
            stds=self.stds,
            weights=weights,
        )

        mode = self.lace_cfg.coherence_mode

        if mode == "none":
            loss_coh = loss_acc.new_tensor(0.0)
            coh_details = {}

        elif mode == "structural":
            model = self._core_model()

            loss_coh, coh_details = structural_coherence_loss(
                mappings=model.effective_mappings(),
                biases=model.effective_biases(),
                hierarchy=self.hierarchy,
                bias_weight=self.lace_cfg.structural_bias_weight,
            )

        else:
            raise RuntimeError(
                f"Unsupported coherence mode: {mode}."
            )

        loss_total = (
            loss_acc
            + self.lace_cfg.lambda_coherence * loss_coh
        )

        details = {
            **acc_details,
            **coh_details,
            "L_accuracy": float(loss_acc.detach().item()),
            "L_coherence": float(loss_coh.detach().item()),
            "L_total": float(loss_total.detach().item()),
        }

        return loss_total, details

    def _structural_pair_keys(self):
        """Keys produced by structural_coherence_loss for adjacent pairs."""
        return [
            f"struct_total_{fine}_{coarse}"
            for fine, coarse in self.hierarchy.adjacent_pairs
        ]

    # ==============================================================
    # Training
    # ==============================================================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        _, val_loader = self._get_data(flag="val")

        # Automatic real-data diagnostics before any optimization.
        self._sanity_check(train_loader)

        # Statistics are computed once from training targets only.
        self.compute_multiscale_statistics(train_loader)

        optimizer = self._select_optimizer()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.patience,
        )

        checkpoint_dir = self.args.checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{setting}.pth",
        )

        # ----------------------------------------------------------
        # Efficiency measurement.
        #
        # Start after setup/statistics so the measurement represents
        # the actual optimization + validation loop.
        # ----------------------------------------------------------
        efficiency_cuda = (
            torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )

        if efficiency_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        train_start = time.perf_counter()
        epochs_run = 0

        for epoch in range(self.args.train_epochs):
            epochs_run = epoch + 1
            self.model.train()

            running_total = 0.0
            running_acc = 0.0
            running_coh = 0.0
            batch_count = 0

            pair_running = {
                    key: 0.0
                    for key in self._structural_pair_keys()
                }

            print(
                f"\n===== EPOCH {epoch + 1}/"
                f"{self.args.train_epochs} ====="
            )

            for i, raw_batch in enumerate(train_loader):
                batch = self._prepare_batch(*raw_batch)

                optimizer.zero_grad(set_to_none=True)

                outputs = self.model(
                    batch["x_enc"],
                    batch["x_mark_enc"],
                    batch["x_dec"],
                    batch["x_mark_dec"],
                )

                loss, details = self._compute_loss(
                    outputs,
                    batch["y_true"],
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch={epoch + 1}, "
                        f"batch={i}: {details}"
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                running_total += details["L_total"]
                running_acc += details["L_accuracy"]
                running_coh += details["L_coherence"]
                batch_count += 1

                for key in pair_running:
                    if key in details:
                        pair_running[key] += details[key]


                if i % 100 == 0:
                    msg = (
                        f"  iter {i:5d} | "
                        f"total={details['L_total']:.6f} | "
                        f"acc={details['L_accuracy']:.6f} | "
                        f"coh={details['L_coherence']:.6f}"
                    )

                    if self.lace_cfg.coherence_mode == "structural":
                        for fine, coarse in self.hierarchy.adjacent_pairs:
                            key = f"struct_total_{fine}_{coarse}"

                            if key in details:
                                msg += (
                                    f" | {fine}->{coarse}="
                                    f"{details[key]:.6f}"
                                )

                    print(msg)

            if batch_count == 0:
                raise RuntimeError("Training loader produced zero batches.")

            train_total = running_total / batch_count
            train_acc = running_acc / batch_count
            train_coh = running_coh / batch_count

            train_pair_avg = {
                    key: value / batch_count
                    for key, value in pair_running.items()
                }

            val_loss, val_details = self.validate(
                val_loader=val_loader,
            )

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[EPOCH {epoch + 1}] "
                f"train_total={train_total:.6f} | "
                f"train_acc={train_acc:.6f} | "
                f"train_coh={train_coh:.6f} | "
                f"val_total={val_loss:.6f} | "
                f"val_acc={val_details['L_accuracy']:.6f} | "
                f"val_coh={val_details['L_coherence']:.6f} | "
                f"lr={current_lr:.3e}"
            )
            if self.lace_cfg.coherence_mode == "structural":
                pair_msg = "  structural pairs:"

                for fine, coarse in self.hierarchy.adjacent_pairs:
                    key = f"struct_total_{fine}_{coarse}"

                    pair_msg += (
                        f" {fine}->{coarse}="
                        f"{train_pair_avg[key]:.6f}"
                    )

                print(pair_msg)

            if self.use_wandb:
                import wandb
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/total": train_total,
                        "train/accuracy": train_acc,
                        "train/coherence": train_coh,
                        "val/total": val_loss,
                        "val/accuracy": val_details["L_accuracy"],
                        "val/coherence": val_details["L_coherence"],
                        "lr": current_lr,
                    }
                )

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.bad_epochs = 0

                torch.save(
                    self.model.state_dict(),
                    checkpoint_path,
                )

                print(
                    f"  saved best checkpoint: "
                    f"{checkpoint_path}"
                )

            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                print(
                    f"EARLY STOPPING after {epoch + 1} epochs."
                )
                break

        if efficiency_cuda:
            torch.cuda.synchronize(self.device)

        train_seconds = (
            time.perf_counter()
            - train_start
        )

        if efficiency_cuda:
            peak_allocated_mb = (
                torch.cuda.max_memory_allocated(self.device)
                / (1024 ** 2)
            )
            peak_reserved_mb = (
                torch.cuda.max_memory_reserved(self.device)
                / (1024 ** 2)
            )
        else:
            peak_allocated_mb = float("nan")
            peak_reserved_mb = float("nan")

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                "Training finished without producing a checkpoint."
            )

        self._append_efficiency_record(
            setting=setting,
            epochs_run=epochs_run,
            train_seconds=train_seconds,
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
        )

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device,
            )
        )

        return self.model

    # ==============================================================
    # Validation
    # ==============================================================

    @torch.no_grad()
    def validate(self, setting=None, val_loader=None):
        if val_loader is None:
            _, val_loader = self._get_data(flag="val")

        was_training = self.model.training
        self.model.eval()

        running_total = 0.0
        running_acc = 0.0
        running_coh = 0.0
        count = 0

        for raw_batch in val_loader:
            batch = self._prepare_batch(*raw_batch)

            outputs = self.model(
                batch["x_enc"],
                batch["x_mark_enc"],
                batch["x_dec"],
                batch["x_mark_dec"],
            )

            loss, details = self._compute_loss(
                outputs,
                batch["y_true"],
            )

            running_total += details["L_total"]
            running_acc += details["L_accuracy"]
            running_coh += details["L_coherence"]
            count += 1

        if count == 0:
            raise RuntimeError(
                "Validation loader produced zero batches."
            )

        result = {
            "L_total": running_total / count,
            "L_accuracy": running_acc / count,
            "L_coherence": running_coh / count,
        }

        if was_training:
            self.model.train()

        return result["L_total"], result

    # ==============================================================
    # Evaluation helpers
    # ==============================================================

    def _inverse_scale(self, values, factor, mean, std):
        """
        Convert aggregated normalized predictions back to physical units.

        Base scaling:
            x = std * x_norm + mean

        Sum hierarchy:
            sum(x) = std * sum(x_norm) + factor * mean

        Mean hierarchy:
            mean(x) = std * mean(x_norm) + mean
        """
        if self.hierarchy.reduction == "sum":
            return std * values + factor * mean

        if self.hierarchy.reduction == "mean":
            return std * values + mean

        raise RuntimeError(
            f"Unsupported hierarchy reduction: "
            f"{self.hierarchy.reduction}."
        )

    # ==============================================================
    # Test
    # ==============================================================

    @torch.no_grad()
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        train_data, train_loader = self._get_data(flag="train")

        checkpoint_path = os.path.join(
            self.args.checkpoints,
            f"{setting}.pth",
        )

        if test:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self.device,
                )
            )

        # Required if test() is called from a fresh experiment object.
        self.compute_multiscale_statistics(train_loader)

        self.model.eval()

        # ----------------------------------------------------------
        # Plot output: same THUML convention as the zero-shot runs.
        # One plot is saved every 20 TEST BATCHES for every factor.
        # ----------------------------------------------------------
        plot_folder = os.path.join(
            "./test_results",
            setting,
        )
        os.makedirs(plot_folder, exist_ok=True)

        # ----------------------------------------------------------
        # Numerical evaluation output.
        #
        # Mirrors the zero-shot documentation:
        #   EVALUATION_RESULTS_PATH / model / data_name /
        #   data_resolution / coherence_mode
        #
        # LACE is multiscale, so factor-specific pred/true files are
        # used to avoid overwriting outputs with different lengths.
        # ----------------------------------------------------------
        data_name = getattr(
            self.args,
            "data_name",
            os.path.splitext(
                os.path.basename(self.args.data_path)
            )[0],
        )

        data_resolution = getattr(
            self.args,
            "data_resolution",
            "multiscale",
        )

        evaluation_folder = os.path.join(
            EVALUATION_RESULTS_PATH,
            self.args.model,
            data_name,
            data_resolution,
            self.lace_cfg.coherence_mode,
        )
        os.makedirs(evaluation_folder, exist_ok=True)

        all_preds = {
            factor: []
            for factor in self.factors
        }

        all_trues = {
            factor: []
            for factor in self.factors
        }

        scaler = train_data.scaler
        base_mean = float(scaler.mean_[0])
        base_std = float(scaler.scale_[0])

        for i, raw_batch in enumerate(test_loader):
            batch = self._prepare_batch(*raw_batch)

            outputs = self.model(
                batch["x_enc"],
                batch["x_mark_enc"],
                batch["x_dec"],
                batch["x_mark_dec"],
            )

            for factor in self.factors:
                pred = outputs[factor]
                true = batch["y_true"][factor]

                all_preds[factor].append(
                    pred.detach().cpu().numpy()
                )

                all_trues[factor].append(
                    true.detach().cpu().numpy()
                )

            # ------------------------------------------------------
            # Documentation plots: every 20 test batches.
            # ------------------------------------------------------
            if i % 20 == 0:
                for factor in self.factors:
                    pred_plot = (
                        outputs[factor][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    true_plot = (
                        batch["y_true"][factor][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if self.args.inverse:
                        pred_plot = self._inverse_scale(
                            pred_plot,
                            factor,
                            base_mean,
                            base_std,
                        )
                        true_plot = self._inverse_scale(
                            true_plot,
                            factor,
                            base_mean,
                            base_std,
                        )

                    visual(
                        true_plot,
                        pred_plot,
                        os.path.join(
                            plot_folder,
                            f"batch_{i}_factor_{factor}.pdf",
                        ),
                    )

        results = {}
        metrics_rows = []

        for factor in self.factors:
            if not all_preds[factor]:
                raise RuntimeError(
                    f"No test predictions collected for factor {factor}."
                )

            preds = np.concatenate(
                all_preds[factor],
                axis=0,
            )

            trues = np.concatenate(
                all_trues[factor],
                axis=0,
            )

            if self.args.inverse:
                preds_eval = self._inverse_scale(
                    preds,
                    factor,
                    base_mean,
                    base_std,
                )
                trues_eval = self._inverse_scale(
                    trues,
                    factor,
                    base_mean,
                    base_std,
                )

            else:
                mu = self.mus[factor]
                sig = self.stds[factor]

                preds_eval = (
                    preds - mu
                ) / (sig + 1e-8)

                trues_eval = (
                    trues - mu
                ) / (sig + 1e-8)

            mae, mse_v, rmse, mape, mspe = metric(
                preds_eval,
                trues_eval,
            )

            results[factor] = {
                "mae": mae,
                "mse": mse_v,
                "rmse": rmse,
                "mape": mape,
                "mspe": mspe,
            }

            metrics_rows.append(
                [mae, mse_v, rmse, mape, mspe]
            )

            print(
                f"[Factor {factor}] "
                f"MSE={mse_v:.6f} | "
                f"MAE={mae:.6f} | "
                f"RMSE={rmse:.6f}"
            )

            # ------------------------------------------------------
            # Factor-specific raw evaluation arrays.
            # ------------------------------------------------------
            np.save(
                os.path.join(
                    evaluation_folder,
                    f"pred_factor_{factor}.npy",
                ),
                preds_eval,
            )

            np.save(
                os.path.join(
                    evaluation_folder,
                    f"true_factor_{factor}.npy",
                ),
                trues_eval,
            )

            if self.use_wandb:
                import wandb
                wandb.log(
                    {
                        f"test/factor_{factor}/mse": mse_v,
                        f"test/factor_{factor}/mae": mae,
                        f"test/factor_{factor}/rmse": rmse,
                    }
                )

        # Same metric order as the zero-shot documentation:
        # [MAE, MSE, RMSE, MAPE, MSPE]
        np.save(
            os.path.join(
                evaluation_folder,
                "metrics.npy",
            ),
            np.asarray(metrics_rows, dtype=np.float64),
        )

        # Row i in metrics.npy corresponds to factors[i].
        np.save(
            os.path.join(
                evaluation_folder,
                "factors.npy",
            ),
            np.asarray(self.factors, dtype=np.int64),
        )

        print(
            "Saved evaluation results to:",
            evaluation_folder,
        )
        print(
            "Saved test plots to:",
            plot_folder,
        )

        result_file = "./result_lace_v2.txt"

        with open(result_file, "a") as f:
            f.write(f"{setting}\n")
            f.write(
                f"coherence_mode="
                f"{self.lace_cfg.coherence_mode}, "
                f"lambda="
                f"{self.lace_cfg.lambda_coherence}\n"
            )
            f.write(
                f"factors={self.factors}\n"
            )

            for factor in self.factors:
                r = results[factor]
                f.write(
                    f"factor {factor}: "
                    f"MSE={r['mse']:.6f}, "
                    f"MAE={r['mae']:.6f}, "
                    f"RMSE={r['rmse']:.6f}\n"
                )

            f.write("\n")

        return results
