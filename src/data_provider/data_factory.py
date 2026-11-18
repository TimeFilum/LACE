"""Load local CSVs with chronological train, validation, and test splits."""
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Fixed_Split


data_dict = {'fixed_split': Dataset_Fixed_Split}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(
            f"Unsupported dataset loader: {args.data!r}. "
            "Use --data fixed_split for all datasets; "
            "select the CSV with --root_path and --data_path."
        )
    DataClass = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    data_set = DataClass(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
        test_split_timestamp=getattr(args, "test_split_timestamp", None),
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False if flag.lower() == 'test' else True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader
