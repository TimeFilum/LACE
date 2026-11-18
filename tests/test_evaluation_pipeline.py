"""Regression checks for baseline exports and reconciliation controls."""
from contextlib import redirect_stdout
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


class EvaluationPipelineTests(unittest.TestCase):
    def test_supervised_training_exports_physical_forecasts_and_bottomup(self):
        from exp import exp_long_term_forecasting_advanced as experiment

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                dates = pd.date_range("2020-01-01", periods=2200, freq="15min")
                values = 20 + 3 * np.sin(np.arange(len(dates)) / 13)
                pd.DataFrame({"date": dates, "value": values}).to_csv(work / "series.csv", index=False)
                for model in ("DLinear", "LSTM"):
                    with self.subTest(model=model):
                        args = SimpleNamespace(
                            model=model, task_name="long_term_forecast",
                            use_gpu=False, gpu_type="cuda", gpu=0, use_multi_gpu=False,
                            seq_len=96, label_len=48, pred_len=96, enc_in=1, dec_in=1,
                            c_out=1, individual=False, moving_avg=25, lstm_hidden=8,
                            lstm_layers=1, embed="timeF", freq="t", dropout=0.0,
                            data="fixed_split", root_path=str(work), data_path="series.csv",
                            target="value", features="S", seasonal_patterns=None,
                            test_split_timestamp=str(dates[1700]), augmentation_ratio=0,
                            batch_size=8, num_workers=0, learning_rate=1e-3,
                            train_epochs=1, patience=2, use_amp=False, use_dtw=False,
                            inverse=True, lradj="type1", data_name="ETTm1_HUFL",
                            data_resolution="15m", model_id=model,
                            checkpoints=str(work / "checkpoints"),
                        )
                        with patch.object(experiment, "EVALUATION_RESULTS_PATH", str(work / "evaluation")), \
                             patch.object(experiment, "EXP_PATH", str(work / "evaluation")), \
                             patch.object(experiment, "RESULTS_PATH", str(work / "results") + os.sep), \
                             patch.object(experiment, "visual"), redirect_stdout(io.StringIO()):
                            exp = experiment.Exp_Long_Term_Forecast(args)
                            provider = exp._get_data
                            cached = {}

                            def bounded_data(flag):
                                if flag not in cached:
                                    dataset, loader = provider(flag)
                                    cached[flag] = dataset, list(itertools.islice(loader, 2))
                                return cached[flag]

                            exp._get_data = bounded_data
                            exp.train(model)
                            exp.test(model)

                        self.assertTrue((work / "checkpoints" / model / "checkpoint.pth").is_file())
                        folder = work / "evaluation" / f"{model}_independent" / "ETTm1_HUFL" / "15m" / "direct"
                        pred, true = np.load(folder / "pred.npy"), np.load(folder / "true.npy")
                        self.assertEqual(pred.shape, (16, 96, 1))
                        self.assertTrue(np.isfinite(pred).all())
                        np.testing.assert_allclose(true[0, :, 0], values[1700:1796], rtol=1e-6)
                        dataset, batches = cached["test"]
                        batch = batches[0]
                        with torch.no_grad():
                            expected = exp.model(batch[0].float(), batch[2].float(), None, None).numpy()
                        expected = dataset.inverse_transform(expected.reshape(-1, 1)).reshape(expected.shape)
                        np.testing.assert_allclose(pred[:len(batch[0])], expected, rtol=1e-6)

                        bottomup = work / "evaluation" / f"{model}_bottomup" / "ETTm1_HUFL" / "15m" / "bottomup"
                        for factor in (1, 4, 24, 96):
                            for label, source in (("pred", pred), ("true", true)):
                                actual = np.load(bottomup / f"{label}_factor_{factor}.npy")
                                expected = source.reshape(len(source), 96 // factor, factor, 1).sum(axis=2)
                                np.testing.assert_allclose(actual, expected)
                        log = pd.read_csv(work / "results" / "comp_efficiency" / f"{model}_epoch_times.csv")
                        self.assertEqual(list(log), ["Model Name", "Epoch", "DataSize", "Epoch Time (s)"])
        finally:
            torch.set_num_threads(previous_threads)

    def test_a0_controls_preserve_truth_and_produce_coherent_forecasts(self):
        from src.evaluator import reconcile_controls_lace as controls

        rng = np.random.default_rng(12)
        factors = (1, 4, 24, 96)
        fine_true = rng.uniform(1, 10, (193, 96))
        fine_pred = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "LACE_moirai1" / "ETTm1_HUFL" / "15m" / "none"
            source.mkdir(parents=True)
            for factor in factors:
                true = fine_true.reshape(193, 96 // factor, factor).sum(axis=2)
                pred = true + rng.normal(size=true.shape)
                if factor == 1:
                    fine_pred = pred
                np.save(source / f"pred_factor_{factor}.npy", pred)
                np.save(source / f"true_factor_{factor}.npy", true)
            with patch.object(controls, "EVALUATION_RESULTS_PATH", str(root)), redirect_stdout(io.StringIO()):
                for method, operation in (("bottomup", controls.make_a0_bottomup),
                                          ("mint_struct", controls.make_a0_wls)):
                    alias = operation("LACE_moirai1", "ETTm1_HUFL", str(root))
                    folder = root / f"{alias}_{method}" / "ETTm1_HUFL" / "15m" / method
                    bottom = np.load(folder / "pred_factor_1.npy")[..., 0]
                    truth = fine_true if method == "bottomup" else fine_true[::96]
                    if method == "bottomup":
                        np.testing.assert_allclose(bottom, fine_pred)
                    for factor in factors:
                        pred = np.load(folder / f"pred_factor_{factor}.npy")[..., 0]
                        true = np.load(folder / f"true_factor_{factor}.npy")[..., 0]
                        expected = bottom.reshape(len(bottom), 96 // factor, factor).sum(axis=2)
                        np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)
                        expected_truth = truth.reshape(len(truth), 96 // factor, factor).sum(axis=2)
                        np.testing.assert_allclose(true, expected_truth)

    def test_supervised_wls_and_ols_exports_are_coherent(self):
        from src.evaluator import reconcile_baselines as reconciliation
        from src.evaluator import evaluate_baseline as evaluation

        rng = np.random.default_rng(20)
        finest = rng.uniform(1, 10, 3 * 96)
        factors = (1, 4, 24, 96)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for factor, resolution in zip(factors, ("15m", "1h", "6h", "1d")):
                series = finest.reshape(-1, factor).sum(axis=1)
                windows = np.lib.stride_tricks.sliding_window_view(series, 96 // factor)
                folder = root / "DLinear_independent" / "ETTm1_HUFL" / resolution / "direct"
                folder.mkdir(parents=True)
                np.save(folder / "true.npy", windows[..., None])
                np.save(folder / "pred.npy", (windows + rng.normal(size=windows.shape))[..., None])
            with patch.object(reconciliation, "EVALUATION_RESULTS_PATH", str(root)), \
                 patch.object(evaluation, "EVALUATION_RESULTS_PATH", str(root)), \
                 redirect_stdout(io.StringIO()):
                for method in ("mint_struct", "ols"):
                    reconciliation.process_dataset("DLinear", "ETTm1_HUFL", method)
                    folder = root / f"DLinear_{method}" / "ETTm1_HUFL" / "15m" / method
                    bottom = np.load(folder / "pred_factor_1.npy")
                    bottom = bottom.reshape(3, 96)
                    for factor in factors:
                        pred = np.load(folder / f"pred_factor_{factor}.npy").reshape(3, 96 // factor)
                        true = np.load(folder / f"true_factor_{factor}.npy").reshape(3, 96 // factor)
                        expected = bottom.reshape(3, 96 // factor, factor).sum(axis=2)
                        np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)
                        expected_truth = finest.reshape(3, 96 // factor, factor).sum(axis=2)
                        np.testing.assert_allclose(true, expected_truth)
                    pairwise, accuracy, _, _ = evaluation.evaluate_baseline(
                        "DLinear", method, "ETTm1_HUFL", road_data_root=str(root)
                    )
                    self.assertEqual(len(accuracy), 4)
                    self.assertTrue(np.isfinite(accuracy["NMSE"]).all())
                    self.assertLess(pairwise["C_rs"].abs().max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
