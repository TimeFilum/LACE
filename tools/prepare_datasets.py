"""Prepare multiresolution CSVs for the study datasets."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def sum_resample(frame, rule, expected_count=None):
    grouped = frame['value'].resample(rule, origin='start_day', label='left', closed='left')
    out = grouped.sum().to_frame('value')
    if expected_count is not None:
        out = out.loc[grouped.count() == expected_count]
    return out


def prepare_electricity(path):
    raw = pd.read_csv(path)
    # Apply the study's HUFL conversion before temporal aggregation.
    finest = pd.DataFrame({'date':pd.to_datetime(raw['date']),
                           'value':raw['HUFL'] * 0.25}).sort_values('date').set_index('date')
    return {
        'ETTm1_HUFL_15m.csv':finest,
        'ETTm1_HUFL_1h.csv':sum_resample(finest,'1h',4),
        'ETTm1_HUFL_6h.csv':sum_resample(finest,'6h',24),
        'ETTm1_HUFL_1d.csv':sum_resample(finest,'1D',96),
    }


def prepare_road(path):
    raw = np.load(path)['data']
    if raw.ndim != 3 or raw.shape[1] <= 240:
        raise ValueError(f'Expected PeMSD4 [time, sensors, channels], got {raw.shape}')
    flow = raw[:,:,0].astype(float)
    flow[flow == 0] = np.nan
    flow = pd.DataFrame(flow).interpolate(method='linear',axis=0,limit_direction='both')
    dates = pd.date_range('2018-01-01 00:00:00',periods=len(flow),freq='5min',name='date')
    outputs = {}
    for sensor in [62,103,240]:
        base = pd.DataFrame({'value':flow[sensor].to_numpy()},index=dates)
        for resolution,rule in [('5m',None),('15m','15min'),('1h','1h'),('1d','1D')]:
            outputs[f'road_sensor_{sensor}_{resolution}.csv'] = base if rule is None else sum_resample(base,rule)
    return outputs


def prepare_network(path, dataset):
    raw = pd.read_csv(path,parse_dates=['date'])
    if 'total_traffic_kbps' not in raw:
        raise ValueError('Input must already contain date and total_traffic_kbps columns.')
    base = raw[['date','total_traffic_kbps']].sort_values('date').set_index('date')
    if base.index.has_duplicates:
        raise ValueError('Duplicate input timestamps.')
    specifications = (
        [('10m',None),('1h','1h'),('6h','6h'),('1d','1D')]
        if dataset=='c_traffic' else
        [('5m',None),('15m','15min'),('1h','1h'),('1d','1D')]
    )
    prefix = 'c' if dataset=='c_traffic' else 'k'
    outputs = {}
    for resolution,rule in specifications:
        outputs[f'{prefix}{resolution}_ideal.csv'] = base if rule is None else base.resample(rule).sum()
    return outputs


def write_outputs(outputs, destination, overwrite=False):
    destination = Path(destination)
    targets = [destination/name for name in outputs]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError('Outputs already exist; use --overwrite explicitly: ' + ', '.join(existing))
    for name,frame in outputs.items():
        if frame.empty or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'Empty or nonfinite prepared data: {name}')
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():
        frame.to_csv(destination/name,index_label='date')
        print(f'{destination/name}: {len(frame)} rows')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['electricity','road','c_traffic','k_traffic'])
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('data'))
    p.add_argument('--overwrite',action='store_true')
    args = p.parse_args()
    if args.dataset=='electricity':
        outputs,directory = prepare_electricity(args.input),'Electricity'
    elif args.dataset=='road':
        outputs,directory = prepare_road(args.input),'Road'
    else:
        outputs,directory = prepare_network(args.input,args.dataset),'Network'
    write_outputs(outputs,args.output_root/directory,args.overwrite)


if __name__=='__main__':
    main()
