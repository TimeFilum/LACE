import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.cuda"
)


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from config import EVALUATION_RESULTS_PATH, RESULTS_PATH, FIG_PATH


EXP_PATH = EVALUATION_RESULTS_PATH
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def get_model_size(self):
    # Access the model directly
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / 1024**2
        return total_size_mb



    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            epoch_duration = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_duration:.2f} seconds")

            # Append epoch time to CSV.
            log_dir = os.path.join(
                RESULTS_PATH,
                "comp_efficiency",
            )

            os.makedirs(
                log_dir,
                exist_ok=True,
            )

            log_path = os.path.join(
                log_dir,
                self.args.model_id + "_epoch_times.csv",
            )

            if epoch == 0 and not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write(
                        "Model Name,Epoch,DataSize,Epoch Time (s)\n"
                    )

            with open(log_path, "a") as f:
                f.write(
                    f"{self.args.model},"
                    f"{epoch + 1},"
                    f"{len(train_data)},"
                    f"{epoch_duration:.2f}\n"
                )

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        model_size_path = os.path.join(RESULTS_PATH + "comp_efficiency", self.args.model_id + '_model_size.csv')

        # Check if file exists before writing header
        write_header = not os.path.exists(model_size_path)

        with open(model_size_path, 'a') as f:
            if write_header:
                f.write('Model Name,DataSize,Model Size (MB)\n')
            f.write(f'{self.args.model},{len(train_data)},{self.get_model_size():.2f}\n')



        return self.model


    def _save_bottomup(self, preds, trues):
        """
        Create Bottom-Up outputs from the finest-resolution baseline run.

        Bottom-Up:
            one finest-resolution forecast
                -> deterministic temporal aggregation
                -> all coarser-resolution forecasts

        Both predictions and truths are aggregated from the same finest
        test windows so that per-resolution error metrics can be computed.
        """

        configs = {
            "c_traffic": {
                "base_resolution": "10m",
                "factors": [1, 6, 36, 144],
            },

            "k_traffic": {
                "base_resolution": "5m",
                "factors": [1, 3, 12, 288],
            },

            "ETTm1_HUFL": {
                "base_resolution": "15m",
                "factors": [1, 4, 24, 96],
            },

            "road_sensor_62": {
                "base_resolution": "5m",
                "factors": [1, 3, 12, 288],
            },

            "road_sensor_103": {
                "base_resolution": "5m",
                "factors": [1, 3, 12, 288],
            },

            "road_sensor_240": {
                "base_resolution": "5m",
                "factors": [1, 3, 12, 288],
            },
        }


        data_name = str(self.args.data_name)
        data_resolution = str(self.args.data_resolution)


        # ----------------------------------------------------------
        # Use the configured study hierarchy.
        # ----------------------------------------------------------

        if data_name not in configs:
            return


        cfg = configs[data_name]


        # ----------------------------------------------------------

        # Bottom-Up must be generated ONLY from the finest run.
        #
        # Coarse-resolution runs remain independent forecasts and will
        # be used for the independent baseline and WLS-Struct.
        # ----------------------------------------------------------

        if data_resolution != cfg["base_resolution"]:
            return


        # ----------------------------------------------------------
        # Expected:
        #     preds : [N, finest_horizon, 1]
        #     trues : [N, finest_horizon, 1]
        # ----------------------------------------------------------

        if preds.ndim != 3 or trues.ndim != 3:
            raise RuntimeError(
                "Bottom-Up expects [N, T, C] arrays, "
                f"got pred={preds.shape}, true={trues.shape}"
            )

        if preds.shape != trues.shape:
            raise RuntimeError(
                "Bottom-Up prediction/truth shape mismatch: "
                f"pred={preds.shape}, true={trues.shape}"
            )


        # ----------------------------------------------------------
        # Exact block-sum aggregation.
        #
        # This matches the current temporal hierarchy:
        #     reduction = sum
        # ----------------------------------------------------------

        def aggregate_blocks(x, factor):

            if factor == 1:
                return x.copy()

            n, t, c = x.shape

            if t % factor != 0:
                raise RuntimeError(
                    f"{data_name}: horizon {t} cannot be divided "
                    f"exactly by factor {factor}."
                )

            return (
                x.reshape(
                    n,
                    t // factor,
                    factor,
                    c,
                )
                .sum(axis=2)
            )


        # ----------------------------------------------------------
        # Output:
        #
        # EVALUATION_RESULTS_PATH/
        #   DLinear_bottomup/
        #     <dataset>/
        #       5m/
        #         bottomup/
        #           pred_factor_1.npy
        #           true_factor_1.npy
        #           ...
        # ----------------------------------------------------------

        folder = os.path.join(
            EVALUATION_RESULTS_PATH,
            f"{self.args.model}_bottomup",
            data_name,
            cfg["base_resolution"],
            "bottomup",
        )

        os.makedirs(
            folder,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # Generate BOTH predictions and truths at every factor.
        # ----------------------------------------------------------

        for factor in cfg["factors"]:

            pred_factor = aggregate_blocks(
                preds,
                factor,
            )

            true_factor = aggregate_blocks(
                trues,
                factor,
            )


            np.save(
                os.path.join(
                    folder,
                    f"pred_factor_{factor}.npy",
                ),
                pred_factor,
            )

            np.save(
                os.path.join(
                    folder,
                    f"true_factor_{factor}.npy",
                ),
                true_factor,
            )


            print(
                f"[BOTTOM-UP] factor={factor} | "
                f"pred={pred_factor.shape} | "
                f"true={true_factor.shape}"
            )


        print(
            f"[BOTTOM-UP] saved to: {folder}"
        )


    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag="test")

        # ============================================================
        # Load checkpoint for test-only execution
        # ============================================================

        if test:
            print("loading model")

            checkpoint_path = os.path.join(
                "./checkpoints",
                setting,
                "checkpoint.pth",
            )

            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self.device,
                )
            )


        # ============================================================
        # FORECAST OUTPUTS
        #
        # EVALUATION_RESULTS_PATH/
        #
        #   DLinear_independent/
        #       <dataset>/
        #           5m/
        #               direct/
        #                   pred.npy
        #                   true.npy
        #                   metrics.npy
        #
        #   DLinear_bottomup/
        #       <dataset>/
        #           5m/
        #               bottomup/
        #                   pred_factor_*.npy
        #                   true_factor_*.npy
        #
        #   test_results/
        #       DLinear/
        #           <dataset>/
        #               5m/
        #                   *.pdf
        #
        #   all_results/
        #       result_<model>_<dataset>_<resolution>.csv
        #
        # ============================================================

        data_name = str(self.args.data_name)
        data_resolution = str(self.args.data_resolution)
        model_name = str(self.args.model)


        # ------------------------------------------------------------
        # Direct independently trained forecast.
        #
        # These are consumed by WLS-Struct reconciliation.
        # ------------------------------------------------------------

        direct_folder = os.path.join(
            EXP_PATH,
            f"{model_name}_independent",
            data_name,
            data_resolution,
            "direct",
        )

        os.makedirs(
            direct_folder,
            exist_ok=True,
        )


        # ------------------------------------------------------------
        # Diagnostic plots
        # ------------------------------------------------------------

        visual_folder = os.path.join(
            EXP_PATH,
            "test_results",
            model_name,
            data_name,
            data_resolution,
        )

        os.makedirs(
            visual_folder,
            exist_ok=True,
        )


        # ============================================================
        # TEST
        # ============================================================

        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = (
                    batch_x_mark
                    .float()
                    .to(self.device)
                )

                batch_y_mark = (
                    batch_y_mark
                    .float()
                    .to(self.device)
                )


                # ----------------------------------------------------
                # Decoder input
                # ----------------------------------------------------

                dec_inp = torch.zeros_like(
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                    ]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[
                            :,
                            :self.args.label_len,
                            :
                        ],
                        dec_inp,
                    ],
                    dim=1,
                ).float().to(self.device)


                # ----------------------------------------------------
                # Forward
                # ----------------------------------------------------

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )

                else:

                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )


                f_dim = (
                    -1
                    if self.args.features == "MS"
                    else 0
                )

                outputs = outputs[
                    :,
                    -self.args.pred_len:,
                    :
                ]

                batch_y = batch_y[
                    :,
                    -self.args.pred_len:,
                    :
                ].to(self.device)


                outputs = (
                    outputs
                    .detach()
                    .cpu()
                    .numpy()
                )

                batch_y = (
                    batch_y
                    .detach()
                    .cpu()
                    .numpy()
                )


                # ----------------------------------------------------
                # Inverse scaling
                # ----------------------------------------------------

                if test_data.scale and self.args.inverse:

                    shape = batch_y.shape

                    if outputs.shape[-1] != batch_y.shape[-1]:

                        outputs = np.tile(
                            outputs,
                            [
                                1,
                                1,
                                int(
                                    batch_y.shape[-1]
                                    / outputs.shape[-1]
                                ),
                            ],
                        )

                    outputs = (
                        test_data
                        .inverse_transform(
                            outputs.reshape(
                                shape[0] * shape[1],
                                -1,
                            )
                        )
                        .reshape(shape)
                    )

                    batch_y = (
                        test_data
                        .inverse_transform(
                            batch_y.reshape(
                                shape[0] * shape[1],
                                -1,
                            )
                        )
                        .reshape(shape)
                    )


                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]


                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)


                # ----------------------------------------------------
                # Diagnostic plots
                # ----------------------------------------------------

                if i % 20 == 0:

                    input_data = (
                        batch_x
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if test_data.scale and self.args.inverse:

                        shape = input_data.shape

                        input_data = (
                            test_data
                            .inverse_transform(
                                input_data.reshape(
                                    shape[0] * shape[1],
                                    -1,
                                )
                            )
                            .reshape(shape)
                        )

                    gt = np.concatenate(
                        (
                            input_data[0, :, -1],
                            true[0, :, -1],
                        ),
                        axis=0,
                    )

                    pd = np.concatenate(
                        (
                            input_data[0, :, -1],
                            pred[0, :, -1],
                        ),
                        axis=0,
                    )

                    visual(
                        gt,
                        pd,
                        os.path.join(
                            visual_folder,
                            f"{i}.pdf",
                        ),
                    )


        # ============================================================
        # COLLECT TEST OUTPUTS
        # ============================================================

        preds = np.concatenate(
            preds,
            axis=0,
        )

        trues = np.concatenate(
            trues,
            axis=0,
        )

        print(
            "test shape:",
            preds.shape,
            trues.shape,
        )


        preds = preds.reshape(
            -1,
            preds.shape[-2],
            preds.shape[-1],
        )

        trues = trues.reshape(
            -1,
            trues.shape[-2],
            trues.shape[-1],
        )

        print(
            "test shape:",
            preds.shape,
            trues.shape,
        )


        # ============================================================
        # SAVE DIRECT FORECASTS IMMEDIATELY
        #
        # Every resolution gets these.
        # These are the independent forecasts required by WLS-Struct.
        # ============================================================

        np.save(
            os.path.join(
                direct_folder,
                "pred.npy",
            ),
            preds,
        )

        np.save(
            os.path.join(
                direct_folder,
                "true.npy",
            ),
            trues,
        )

        print(
            f"[DIRECT SAVE] "
            f"{model_name} | "
            f"{data_name} | "
            f"{data_resolution} | "
            f"pred={preds.shape} | "
            f"true={trues.shape}"
        )


        # ============================================================
        # BOTTOM-UP
        #
        # The helper itself checks whether this is the finest
        # resolution. Therefore coarse independent runs do nothing
        # here.
        # ============================================================

        if model_name in [
            "DLinear",
            "LSTM",
        ]:

            self._save_bottomup(
                preds=preds,
                trues=trues,
            )


        # ============================================================
        # DTW
        # ============================================================

        if self.args.use_dtw:

            dtw_list = []

            manhattan_distance = (
                lambda x, y: np.abs(x - y)
            )

            for i in range(preds.shape[0]):

                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)

                if i % 100 == 0:
                    print(
                        "calculating dtw iter:",
                        i,
                    )

                d, _, _, _ = accelerated_dtw(
                    x,
                    y,
                    dist=manhattan_distance,
                )

                dtw_list.append(d)

            dtw = np.array(
                dtw_list
            ).mean()

        else:

            dtw = "Not calculated"


        # ============================================================
        # ERROR METRICS
        # ============================================================

        mae, mse, rmse, mape, mspe = metric(
            preds,
            trues,
        )

        print(
            "mse:{}, mae:{}, dtw:{}".format(
                mse,
                mae,
                dtw,
            )
        )


        # ------------------------------------------------------------
        # Save metrics beside direct prediction/truth.
        # ------------------------------------------------------------

        np.save(
            os.path.join(
                direct_folder,
                "metrics.npy",
            ),
            np.array(
                [
                    mae,
                    mse,
                    rmse,
                    mape,
                    mspe,
                ]
            ),
        )


        # ============================================================
        # SUMMARY CSV
        # ============================================================

        results_dir = os.path.join(
            EXP_PATH,
            "all_results",
        )

        os.makedirs(
            results_dir,
            exist_ok=True,
        )


        file_path = os.path.join(
            results_dir,
            (
                f"result_"
                f"{model_name}_"
                f"{data_name}_"
                f"{data_resolution}.csv"
            ),
        )


        write_header = (
            not os.path.exists(file_path)
            or os.path.getsize(file_path) == 0
        )


        with open(
            file_path,
            "a",
        ) as f:

            if write_header:

                f.write(
                    "Model,MSE,MAE,MAPE,MSPE,RMSE\n"
                )

            f.write(
                "{},{},{},{},{},{}\n".format(
                    model_name,
                    mse,
                    mae,
                    mape,
                    mspe,
                    rmse,
                )
            )


        print(
            f"[TEST COMPLETE] "
            f"{model_name} | "
            f"{data_name} | "
            f"{data_resolution}"
        )

        return
