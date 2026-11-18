
"""Post-processing controls for LACE: zero-shot -> BU/WLS and A0 -> BU/WLS."""
from __future__ import annotations

import argparse
import importlib
import os
import numpy as np

try:
    from src.config import EVALUATION_RESULTS_PATH
except ImportError:
    from config import EVALUATION_RESULTS_PATH
try:
    from src.evaluator.evaluate_lace import get_dataset_config
except ImportError:
    from evaluator.evaluate_lace import get_dataset_config

DATASETS = [
    "c_traffic", "k_traffic", "ETTm1_HUFL",
    "road_sensor_62", "road_sensor_103", "road_sensor_240",
]
PHASE0_MODELS = ["Moirai1", "Moirai2", "Chronos2", "TimesFM2"]
A0_MODELS = ["LACE_moirai1", "LACE_moirai2", "LACE_chronos2", "LACE_timesfm2"]
A0_ALIASES = {
    "LACE_moirai1": "LACE_A0_Moirai1",
    "LACE_moirai2": "LACE_A0_Moirai2",
    "LACE_chronos2": "LACE_A0_Chronos2",
    "LACE_timesfm2": "LACE_A0_TimesFM2",
}


def as2(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected [N,H] or [N,H,1], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values found")
    return x


def as3(x):
    return as2(x)[..., None]


def aggregate_sum(x, factor):
    x = as2(x)
    factor = int(factor)
    if factor == 1:
        return x.copy()
    n, h = x.shape
    if h % factor:
        raise ValueError(f"Horizon {h} not divisible by factor {factor}")
    return x.reshape(n, h // factor, factor).sum(axis=2)


def aggregation_matrix(base_horizon, factor):
    if base_horizon % factor:
        raise ValueError(f"{base_horizon} not divisible by {factor}")
    out = np.zeros((base_horizon // factor, base_horizon), dtype=np.float64)
    for i in range(out.shape[0]):
        out[i, i * factor:(i + 1) * factor] = 1.0
    return out


def summing_matrix(base_horizon, factors):
    return np.concatenate(
        [aggregation_matrix(base_horizon, f) for f in factors], axis=0
    )


def reconcile_wls_struct(base_forecasts, S):
    # Exact match to the existing WLS-Struct convention: W = diag(S @ 1).
    structural_scale = S @ np.ones(S.shape[1], dtype=np.float64)
    W = np.diag(structural_scale)
    scale = max(np.trace(W) / W.shape[0], 1.0)
    W = W + 1e-8 * scale * np.eye(W.shape[0])
    WinvS = np.linalg.solve(W, S)
    G = np.linalg.pinv(S.T @ WinvS) @ WinvS.T
    bottom = base_forecasts @ G.T
    return bottom @ S.T


def stack_levels(xs):
    return np.concatenate([as2(x) for x in xs], axis=1)


def split_levels(x, horizons):
    out, start = [], 0
    for h in horizons:
        out.append(x[:, start:start + h])
        start += h
    if start != x.shape[1]:
        raise RuntimeError("Hierarchy split did not consume all nodes")
    return out


def verify_truth(truths, factors):
    finest = as2(truths[0])
    for direct, factor in zip(truths, factors):
        generated = aggregate_sum(finest, factor)
        err = np.max(np.abs(generated - as2(direct)))
        print(f"[TRUTH] factor={factor:<4} max_abs={err:.3e}")
        if not np.allclose(generated, direct, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"Truth hierarchy failed at factor={factor}")


def verify_reconciled(rec, S, base_horizon):
    reconstructed = rec[:, :base_horizon] @ S.T
    err = np.max(np.abs(rec - reconstructed))
    print(f"[COHERENCE] max_abs={err:.3e}")
    if not np.allclose(rec, reconstructed, rtol=1e-7, atol=1e-7):
        raise RuntimeError("WLS result is not numerically coherent")


def cfg_meta(dataset, road_root):
    try:
        cfg = get_dataset_config(dataset, road_data_root=road_root)
    except TypeError:
        cfg = get_dataset_config(dataset)
    factors = [int(f) for f in cfg["factor_to_res"]]
    resolutions = [cfg["factor_to_res"][f] for f in cfg["factor_to_res"]]
    base_horizon = int(cfg["daily_stride"])
    horizons = [base_horizon // f for f in factors]
    return cfg, factors, resolutions, horizons, base_horizon


def phase0_root(cli_root):
    if cli_root:
        return cli_root
    for name in ("src.config", "config"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        root = getattr(mod, "RESULT_PHASE0_DIR", None)
        if root:
            return str(root)
    raise RuntimeError("RESULT_PHASE0_DIR not found; pass --phase0-root")


def load_phase0(root, model, dataset, resolution):
    folder = os.path.join(root, model, dataset, resolution)
    pred = as2(np.load(os.path.join(folder, "pred.npy")))
    true = as2(np.load(os.path.join(folder, "true.npy")))
    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: {model}/{dataset}/{resolution}")
    return pred, true


def save_factor_outputs(folder, factors, preds, truths):
    os.makedirs(folder, exist_ok=True)
    for f, p, t in zip(factors, preds, truths):
        np.save(os.path.join(folder, f"pred_factor_{f}.npy"), as3(p))
        np.save(os.path.join(folder, f"true_factor_{f}.npy"), as3(t))
        print(f"[SAVE] factor={f:<4} pred={as2(p).shape}")


def make_phase0_bottomup(root, model, dataset, road_root):
    cfg, factors, _, _, _ = cfg_meta(dataset, road_root)
    pred, true = load_phase0(root, model, dataset, cfg["base_resolution"])
    preds = [aggregate_sum(pred, f) for f in factors]
    truths = [aggregate_sum(true, f) for f in factors]
    folder = os.path.join(
        EVALUATION_RESULTS_PATH, f"{model}_bottomup", dataset,
        cfg["base_resolution"], "bottomup"
    )
    # Save ALL sliding finest-resolution origins. Existing evaluator applies
    # the common base daily stride to Bottom-Up arrays.
    save_factor_outputs(folder, factors, preds, truths)
    return model


def make_phase0_wls(root, model, dataset, road_root):
    cfg, factors, resolutions, horizons, base_horizon = cfg_meta(dataset, road_root)
    preds, truths = [], []
    for f, res, horizon in zip(factors, resolutions, horizons):
        p, t = load_phase0(root, model, dataset, res)
        # Independent zero-shot series advances at its native resolution.
        preds.append(p[::horizon])
        truths.append(t[::horizon])
    n = min(len(x) for x in preds)
    preds, truths = [x[:n] for x in preds], [x[:n] for x in truths]
    print(f"[ALIGN] zero-shot common daily origins={n}")
    verify_truth(truths, factors)
    S = summing_matrix(base_horizon, factors)
    rec = reconcile_wls_struct(stack_levels(preds), S)
    verify_reconciled(rec, S, base_horizon)
    rec_levels = split_levels(rec, horizons)
    folder = os.path.join(
        EVALUATION_RESULTS_PATH, f"{model}_mint_struct", dataset,
        cfg["base_resolution"], "mint_struct"
    )
    # WLS arrays are already daily-origin aligned.
    save_factor_outputs(folder, factors, rec_levels, truths)
    return model


def load_a0(model_dir, dataset, base_resolution, factor):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH, model_dir, dataset, base_resolution, "none"
    )
    p = as2(np.load(os.path.join(folder, f"pred_factor_{factor}.npy")))
    t = as2(np.load(os.path.join(folder, f"true_factor_{factor}.npy")))
    if p.shape != t.shape:
        raise RuntimeError(f"A0 shape mismatch: {model_dir}/{dataset}/factor={factor}")
    return p, t


def make_a0_bottomup(model_dir, dataset, road_root):
    """
    Bottom-Up control applied to the already-trained LACE A0 model.

    Uses ONLY the finest A0 forecast, then sum-aggregates that forecast
    upward to every coarser hierarchy level. No retraining is performed.

    Arrays are saved at all sliding finest-resolution origins so the
    existing baseline evaluator applies the usual common daily stride.
    """
    cfg, factors, _, _, _ = cfg_meta(dataset, road_root)

    finest_factor = factors[0]
    if finest_factor != 1:
        raise RuntimeError(
            f"Expected finest hierarchy factor 1, got {finest_factor}"
        )

    fine_pred, fine_true = load_a0(
        model_dir,
        dataset,
        cfg["base_resolution"],
        finest_factor,
    )

    preds = [
        aggregate_sum(fine_pred, f)
        for f in factors
    ]
    truths = [
        aggregate_sum(fine_true, f)
        for f in factors
    ]

    alias = A0_ALIASES[model_dir]

    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{alias}_bottomup",
        dataset,
        cfg["base_resolution"],
        "bottomup",
    )

    save_factor_outputs(
        folder,
        factors,
        preds,
        truths,
    )

    return alias


def make_a0_wls(model_dir, dataset, road_root):
    cfg, factors, _, horizons, base_horizon = cfg_meta(dataset, road_root)
    stride = int(cfg["daily_stride"])
    preds, truths = [], []
    for f in factors:
        p, t = load_a0(model_dir, dataset, cfg["base_resolution"], f)
        # All LACE heads share the same finest-resolution rolling origins.
        preds.append(p[::stride])
        truths.append(t[::stride])
    n = min(len(x) for x in preds)
    preds, truths = [x[:n] for x in preds], [x[:n] for x in truths]
    print(f"[ALIGN] A0 common daily origins={n}")
    verify_truth(truths, factors)
    S = summing_matrix(base_horizon, factors)
    rec = reconcile_wls_struct(stack_levels(preds), S)
    verify_reconciled(rec, S, base_horizon)
    rec_levels = split_levels(rec, horizons)
    alias = A0_ALIASES[model_dir]
    folder = os.path.join(
        EVALUATION_RESULTS_PATH, f"{alias}_mint_struct", dataset,
        cfg["base_resolution"], "mint_struct"
    )
    save_factor_outputs(folder, factors, rec_levels, truths)
    return alias


def evaluate(model, baseline, dataset, road_root):
    try:
        ev = importlib.import_module("src.evaluator.evaluate_baseline")
    except ImportError:
        ev = importlib.import_module("evaluator.evaluate_baseline")
    # Existing evaluator uses SUPPORTED_MODELS only as an input guard.
    if model not in ev.SUPPORTED_MODELS:
        ev.SUPPORTED_MODELS.append(model)
    ev.evaluate_baseline(
        model=model, baseline=baseline, dataset=dataset,
        road_data_root=road_root,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["phase0", "a0"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", choices=["bottomup", "mint_struct"])
    ap.add_argument("--phase0-root", default=None)
    ap.add_argument("--road-data-root", default="./data/Road")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    if args.source == "phase0":
        models = args.models or PHASE0_MODELS
        methods = args.methods or ["bottomup", "mint_struct"]
        root = phase0_root(args.phase0_root)
        bad = set(models) - set(PHASE0_MODELS)
        if bad:
            raise ValueError(f"Unknown phase0 models: {sorted(bad)}")
        for model in models:
            for dataset in args.datasets:
                for method in methods:
                    print("\n" + "=" * 76)
                    print(f"PHASE0 | {model} | {dataset} | {method}")
                    print("=" * 76)
                    if method == "bottomup":
                        label = make_phase0_bottomup(root, model, dataset, args.road_data_root)
                    else:
                        label = make_phase0_wls(root, model, dataset, args.road_data_root)
                    if args.evaluate:
                        evaluate(label, method, dataset, args.road_data_root)
    else:
        models = args.models or A0_MODELS
        methods = args.methods or ["mint_struct"]

        bad = set(models) - set(A0_MODELS)
        if bad:
            raise ValueError(f"Unknown A0 model dirs: {sorted(bad)}")

        for model_dir in models:
            for dataset in args.datasets:
                for method in methods:
                    print("\n" + "=" * 76)
                    print(f"A0 | {model_dir} | {dataset} | {method}")
                    print("=" * 76)

                    if method == "bottomup":
                        alias = make_a0_bottomup(
                            model_dir,
                            dataset,
                            args.road_data_root,
                        )
                    elif method == "mint_struct":
                        alias = make_a0_wls(
                            model_dir,
                            dataset,
                            args.road_data_root,
                        )
                    else:
                        raise ValueError(method)

                    if args.evaluate:
                        evaluate(
                            alias,
                            method,
                            dataset,
                            args.road_data_root,
                        )


if __name__ == "__main__":
    main()
