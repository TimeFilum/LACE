# Dataset preparation

Obtain the source data below and place the prepared CSVs in the repository-root
`data/` directory. Each CSV needs a `date` column and the target column listed in
the experiment settings. Keep rows in chronological order and use timestamps
and timezones consistent with the test boundary.

All experiment launchers use the shared local-CSV loader, `--data fixed_split`.
LACE constructs its targets from the finest-resolution CSV. Independent
forecasts and supervised baselines also use the corresponding coarse-resolution
CSVs. The paper uses sum aggregation.

## Sources

| Dataset | Source |
|---|---|
| CESNET-TimeSeries24 | [Zenodo record](https://zenodo.org/records/13382427), DOI [10.5281/zenodo.13382427](https://doi.org/10.5281/zenodo.13382427). Dataset description: Koumar et al., *Scientific Data* 12, 338 (2025), [10.1038/s41597-025-04603-x](https://doi.org/10.1038/s41597-025-04603-x). |
| ETTm1 | [ETTDataset repository](https://github.com/zhouhaoyi/ETDataset), file `ETT-small/ETTm1.csv`. |
| PeMSD4 | [Caltrans PeMS](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source); benchmark reference: Ma et al., *Spatio-temporal adaptive graph convolutional networks for traffic flow forecasting* (2023). The preparation utility reads the benchmark's `pemsd4.npz` array. |
| Campus traffic (K-traffic) | Collected locally by the authors. The data remain private and are excluded from this release. |

## Experiment settings

| Dataset | Directory | Files | Target | Factors | Test starts at |
|---|---|---|---|---|---|
| CESNET (C-traffic) | `data/Network/` | `c10m_ideal.csv`, `c1h_ideal.csv`, `c6h_ideal.csv`, `c1d_ideal.csv` | `total_traffic_kbps` | 1, 6, 36, 144 | 2024-04-07 00:00:00+0000 |
| Campus (K-traffic) | `data/Network/` | `k5m_ideal.csv`, `k15m_ideal.csv`, `k1h_ideal.csv`, `k1d_ideal.csv` | `total_traffic_kbps` | 1, 3, 12, 288 | 2025-07-01 00:00:00+0000 |
| ETTm1 HUFL | `data/Electricity/` | `ETTm1_HUFL_15m.csv`, `ETTm1_HUFL_1h.csv`, `ETTm1_HUFL_6h.csv`, `ETTm1_HUFL_1d.csv` | `value` | 1, 4, 24, 96 | 2018-01-31 00:00:00 |
| Road sensors 62, 103, 240 | `data/Road/` | `road_sensor_<id>_5m.csv`, `_15m.csv`, `_1h.csv`, `_1d.csv` | `value` | 1, 3, 12, 288 | 2018-02-17 00:00:00 |

## Prepare the CSVs

Use the preparation utility to create the filenames expected by the launchers:

```bash
python tools/prepare_datasets.py --dataset electricity --input /path/to/ETTm1.csv
python tools/prepare_datasets.py --dataset road --input /path/to/pemsd4.npz
python tools/prepare_datasets.py --dataset c_traffic --input /path/to/c10m_ideal.csv
```

Run these commands from the repository root. Use `--output-root` to choose a
different data directory and `--overwrite` to replace existing prepared files.

**Electricity:** select HUFL from `ETTm1.csv`, multiply it by `0.25`, and construct
hourly, six-hourly, and daily sums. Retain intervals with the expected number of
finest-resolution observations.

**Road traffic:** read channel `0` from the PeMSD4 array and select sensor indices
`62`, `103`, and `240`. Treat zeros as missing and apply linear interpolation.
Use five-minute timestamps starting at `2018-01-01 00:00:00`, then construct
15-minute, hourly, and daily sums.

**Network traffic:** start with a prepared finest-resolution CSV containing
`date` and `total_traffic_kbps`. The utility sorts the timestamps and constructs
the coarser series through temporal sum aggregation.

For CESNET, the input is `c10m_ideal.csv`. Reconstructing the study's selected
series from the raw Zenodo archives also requires the original institution,
subnet, or IP selection and field-conversion details. The network preparation
code covers aggregation from the prepared finest-resolution series.

After preparing ETTm1, run a single LACE experiment with:

```bash
ONLY_DATASET=ETTm1_HUFL ONLY_MODE=structural MODEL_NAME=LACE_moirai1 bash scripts/run_lace_overnight.sh
```

## Preparation notebooks

The notebooks illustrate the data transformations and multiresolution
construction:

| Notebook | Contents |
|---|---|
| `notebooks/prepare_electricity_dataset.ipynb` | ETTm1 HUFL selection and temporal aggregation. |
| `notebooks/prepare_road_datasets.ipynb` | PeMSD4 channel and sensor selection, interpolation, and temporal aggregation. |
| `notebooks/prepare_multiscale_network_data.ipynb` | Network-series aggregation and alignment. |

The notebooks walk through the same transformations and write the filenames
expected by the experiment launchers.

## Splits and normalization

Test forecasts begin at the first timestamp at or after the configured test
boundary. The final `10%` of the preceding observations form the validation
period; the rest form training. Validation and test input windows include the
preceding context needed to forecast their targets.

Fit `StandardScaler` on the training period. Construct LACE's coarse targets by
aggregating the transformed finest-resolution future windows. Compute the
per-resolution accuracy-loss means and standard deviations from training
targets as well.
