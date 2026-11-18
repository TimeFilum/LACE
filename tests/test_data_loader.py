"""Release CSV boundary, scaling, and batch-order regression checks."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Fixed_Split


class DataLoaderTests(unittest.TestCase):
    def test_fixed_boundary_and_scaler_use_only_training_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            dates = pd.date_range('2020-01-01', periods=200, freq='h')
            values = np.arange(200, dtype=float)
            pd.DataFrame({'date': dates, 'value': values}).to_csv(
                Path(directory) / 'series.csv', index=False)
            args = SimpleNamespace(
                data='fixed_split', root_path=directory, data_path='series.csv',
                seq_len=12, label_len=6, pred_len=4, features='S', target='value',
                embed='timeF', freq='h', seasonal_patterns=None,
                test_split_timestamp=str(dates[150]), augmentation_ratio=0,
                batch_size=7, num_workers=0,
            )
            # Timestamp split: 135 training rows, 15 validation rows, 50 test rows.
            expected = {'train': (0, 135, 120), 'val': (123, 150, 12),
                        'test': (138, 200, 47)}
            with redirect_stdout(io.StringIO()):
                for flag, (start, end, count) in expected.items():
                    with self.subTest(flag=flag):
                        dataset, loader = data_provider(args, flag)
                        self.assertEqual(len(dataset), count)
                        np.testing.assert_allclose(dataset.scaler.mean_, [67.0])
                        np.testing.assert_allclose(dataset.scaler.scale_, [values[:135].std()])
                        np.testing.assert_allclose(
                            dataset.inverse_transform(dataset.data_x)[:, 0],
                            values[start:end], atol=1e-12)
                        # The first forecast in val/test begins at its split boundary.
                        target = dataset.inverse_transform(dataset[0][1])[:, 0]
                        np.testing.assert_allclose(
                            target[-4:], values[start + 12:start + 16], atol=1e-12)
                        self.assertIsInstance(loader.sampler,
                                              SequentialSampler if flag == 'test' else RandomSampler)
                        self.assertEqual(loader.batch_size, 7)
                        self.assertFalse(loader.drop_last)
                        batches = list(loader)
                        self.assertEqual(sum(len(batch[0]) for batch in batches), count)

    def test_missing_csv_reports_preparation_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, r'docs/DATASETS\.md'):
                Dataset_Fixed_Split(SimpleNamespace(), directory,
                                    data_path='missing.csv', size=[12, 6, 4])

    def test_unsupported_loader_reports_shared_csv_interface(self):
        with self.assertRaisesRegex(ValueError, '--data fixed_split'):
            data_provider(SimpleNamespace(data='m4'), 'train')


if __name__ == '__main__':
    unittest.main()
