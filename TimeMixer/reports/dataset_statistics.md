## TimeMixer – Dataset Statistics

This document summarises the eight benchmark datasets used in the long-term forecasting experiments.
All values are derived from the raw CSV/TXT files in `../dataset/` and the split logic in
`data_provider/data_loader.py`.

---

### Quick-reference Table

| Dataset       | Domain              | File Size | Total Rows | Features | Temporal Resolution | Time Span       | Loader Class        |
|---------------|---------------------|-----------|------------|----------|---------------------|-----------------|---------------------|
| ETTh1         | Electricity (xfmr)  | 1.7 MB    | 17,420     | 7        | 1 hour              | ~2 years        | Dataset_ETT_hour    |
| ETTh2         | Electricity (xfmr)  | 2.4 MB    | 17,421     | 7        | 1 hour              | ~2 years        | Dataset_ETT_hour    |
| ETTm1         | Electricity (xfmr)  | 9.9 MB    | 69,681     | 7        | 15 min              | ~2 years        | Dataset_ETT_minute  |
| ETTm2         | Electricity (xfmr)  | 9.3 MB    | 69,681     | 7        | 15 min              | ~2 years        | Dataset_ETT_minute  |
| Electricity   | Power consumption   | 92 MB     | 26,305     | 321      | 1 hour              | ~3 years        | Dataset_Custom      |
| Weather       | Meteorology         | 7.0 MB    | 52,697     | 21       | 10 min              | ~1 year         | Dataset_Custom      |
| Traffic       | Road occupancy      | 131 MB    | 17,545     | 862      | 1 hour              | ~2 years        | Dataset_Custom      |
| Solar-Energy  | PV generation       | 172 MB    | 52,560     | 137      | 10 min              | ~1 year         | Dataset_Solar       |

---

### Train / Validation / Test Splits

#### ETT datasets (fixed calendar split)

The ETT loaders hard-code splits by calendar months (30-day months assumed):

| Split | Formula (hourly) | Rows | Formula (15-min) | Rows |
|-------|-----------------|------|-----------------|------|
| Train | `0 → 12×30×24` | 8,640 | `0 → 12×30×24×4` | 34,560 |
| Val   | `8,640 → 11,520` | 2,880 | `34,560 → 46,080` | 11,520 |
| Test  | `11,520 → 14,400` | 2,880 | `46,080 → 57,600` | 11,520 |

> The val and test windows are slid back by `seq_len` (96) at their start so that every
> sample has a full look-back context; the effective data coverage is unchanged.

| Dataset | Train rows | Val rows | Test rows | Total used |
|---------|-----------|---------|----------|------------|
| ETTh1   | 8,640     | 2,880   | 2,880    | 14,400     |
| ETTh2   | 8,640     | 2,880   | 2,880    | 14,400     |
| ETTm1   | 34,560    | 11,520  | 11,520   | 57,600     |
| ETTm2   | 34,560    | 11,520  | 11,520   | 57,600     |

#### Custom datasets (proportional split: 70 / 10 / 20 %)

`num_train = int(N × 0.7)`, `num_test = int(N × 0.2)`, `num_val = N − train − test`

| Dataset     | N      | Train  | Val   | Test   |
|-------------|--------|--------|-------|--------|
| Electricity | 26,305 | 18,413 | 2,631 | 5,261  |
| Weather     | 52,697 | 36,887 | 5,271 | 10,539 |
| Traffic     | 17,545 | 12,281 | 1,755 | 3,509  |
| Solar       | 52,560 | 36,792 | 5,256 | 10,512 |

---

### Per-dataset Details

#### ETTh1 & ETTh2

- **Path:** `../dataset/ETT-small/ETTh1.csv` / `ETTh2.csv`
- **Start date:** 2016-07-01 00:00
- **Features (7):** `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`
  - HUFL/HULL: high-useful / high-useless load
  - MUFL/MULL: mid-useful / mid-useless load
  - LUFL/LULL: low-useful / low-useless load
  - OT: oil temperature (target)
- **Normalisation:** StandardScaler fit on train split only
- **TimeMixer config:** `enc_in=7`, `c_out=7`, `d_model=16`, `d_ff=32`, `e_layers=2`

#### ETTm1 & ETTm2

- **Path:** `../dataset/ETT-small/ETTm1.csv` / `ETTm2.csv`
- **Start date:** 2016-07-01 00:00
- **Features (7):** same columns as ETTh (finer temporal resolution)
- **TimeMixer config:** `enc_in=7`, `c_out=7`, `d_model=16`, `d_ff=32`, `e_layers=2`

#### Electricity (ECL)

- **Path:** `../dataset/electricity/electricity.csv`
- **Features (321):** `MT_001 … MT_320, OT` – hourly power consumption of 321 clients (Portugal)
- **TimeMixer config:** `enc_in=321`, `c_out=321`, `d_model=16`, `d_ff=32`, `e_layers=2`

#### Weather

- **Path:** `../dataset/weather/weather.csv`
- **Start date:** 2020-01-01 00:10
- **Features (21):** pressure, temperature (°C & K & dewpoint), relative humidity, vapour pressure
  (max/actual/deficit), specific humidity, water-vapour concentration, air density, wind speed
  (mean & max), wind direction, rainfall, rain duration, shortwave radiation, PAR (mean & max),
  Logger temperature, OT
- **TimeMixer config:** `enc_in=21`, `c_out=21`, `d_model=16`, `d_ff=32`, `e_layers=2`

#### Traffic

- **Path:** `../dataset/traffic/traffic.csv`
- **Start date:** 2016-07-01 02:00
- **Features (862):** road occupancy rates measured by 862 sensors on San Francisco Bay Area
  freeways (Caltrans PeMS); `OT` is the last column
- **TimeMixer config:** `enc_in=862`, `c_out=862`, `d_model=32`, `d_ff=32`, `e_layers=2`

#### Solar-Energy

- **Path:** `../dataset/solar/solar_AL.txt`
- **Format:** headerless comma-separated text; 137 numerical columns
- **Features (137):** solar-power output of 137 PV plants in Alabama (NREL); 10-min intervals,
  52,560 rows = 365 days × 144 records/day (1 year, 2006)
- **TimeMixer config:** `enc_in=137`, `c_out=137`, `d_model=512`, `d_ff=2048`, `e_layers=2`
  (larger model capacity due to high inter-series correlation in solar data)

---

### Model Input/Output Dimensions at a Glance

| Dataset     | enc_in | c_out | seq_len | pred_lens          | d_model | d_ff  |
|-------------|--------|-------|---------|--------------------|---------|-------|
| ETTh1/h2    | 7      | 7     | 96      | 96, 192, 336, 720  | 16      | 32    |
| ETTm1/m2    | 7      | 7     | 96      | 96, 192, 336, 720  | 16      | 32    |
| Electricity | 321    | 321   | 96      | 96, 192, 336, 720  | 16      | 32    |
| Weather     | 21     | 21    | 96      | 96, 192, 336, 720  | 16      | 32    |
| Traffic     | 862    | 862   | 96      | 96, 192, 336, 720  | 32      | 32    |
| Solar       | 137    | 137   | 96      | 96, 192, 336, 720  | 512     | 2048  |

All runs use `down_sampling_layers=3`, `down_sampling_method=avg`, `down_sampling_window=2`,
`batch_size=128 (32 for Traffic/Solar)`, `learning_rate=0.01`, `train_epochs=10`, `patience=10`.
