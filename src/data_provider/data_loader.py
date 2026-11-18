"""Shared local-CSV loader used by all released forecasting experiments."""
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features


class Dataset_Fixed_Split(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None,
                 test_split_timestamp=None,
                 val_ratio=0.1):

        self.args = args
        self.test_split_timestamp = test_split_timestamp
        self.val_ratio = val_ratio

        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        if not os.path.isfile(local_fp):
            raise FileNotFoundError(
                f"Dataset CSV not found: {local_fp}. "
                "See docs/DATASETS.md for sources and preparation instructions."
            )
        df_raw = pd.read_csv(local_fp)

        # Ensure datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Reorder columns: date, features..., target
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # ---------------------------------------------------------
        # TIMESTAMP-BASED SPLITTING
        # ---------------------------------------------------------
        if self.test_split_timestamp is not None:
            ts = pd.to_datetime(self.test_split_timestamp)

            # test starts at first row >= timestamp
            test_start_idx = df_raw.index[df_raw['date'] >= ts][0]

            # train+val ends right before test
            trainval_end = test_start_idx

            num_trainval = trainval_end
            num_val = int(num_trainval * self.val_ratio)
            num_train = num_trainval - num_val

            border1s = [
                0,                          # train
                num_train - self.seq_len,   # val
                test_start_idx - self.seq_len  # test
            ]
            border2s = [
                num_train,                  # train
                trainval_end,               # val
                len(df_raw)                 # test
            ]

        else:
            # ---------------------------------------------------------
            # ORIGINAL RATIO-BASED SPLITTING
            # ---------------------------------------------------------
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select features
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        # Scaling
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time features
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            dates = pd.to_datetime(df_stamp['date'])
            dates = pd.DatetimeIndex(dates)
            data_stamp = time_features(dates, freq=self.freq).transpose(1, 0)


        # Final slices
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Optional augmentation
        if self.set_type == 0 and getattr(self.args, "augmentation_ratio", 0) > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
