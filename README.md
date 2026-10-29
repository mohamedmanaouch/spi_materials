# spi_materials

Supplementary materials for the article **"Multi-scale analysis of meteorological drought: a case study in the Middle Draa Valley, southeastern Morocco"** (Ecological Engineering & Environmental Technology, manuscript EEET-05520-2026).

All the results in the article (Tables 1, 3, 4 and 5, the drought intensities, the Mann–Kendall and Sen results, and the data of Figures 6–9) can be reproduced from the monthly precipitation series with one command.

## Files

| File | Content |
|---|---|
| `S1_Monthly_rainfall/` | Monthly precipitation totals (mm), one Excel workbook per station (Agdez, Zagora, Tagounite, M'hamid El Ghizlane): year, month, total. January 1989 to December 2020, no missing month (the Agdez workbook runs to December 2022; the script ignores the months after 2020). Source: Agricultural Subdivision of Zagora (ORMVAO). |
| `S1_monthly_precipitation_1989-2020.csv` | The same 1989–2020 data in a single CSV file (identical results). |
| `S2_SPI_monthly_series_1989-2020.csv` | Monthly SPI-3, SPI-6, SPI-9 and SPI-12 of the four stations, produced by `S3` from `S1`. These are the values used in the article. |
| `S3_spi_middle_draa.py` | Python script (version 1.1) that computes the SPI and all drought statistics. |
| `S4_SPI_Generator_series_1989-2020.csv` | SPI series first computed with SPI Generator (National Drought Mitigation Center), kept for the cross-check described in the article. |
| `requirements.txt` | Python packages. |

The daily records from which the monthly totals were derived are held by ORMVAO and are not distributed here.

## Reproduce the results

```bash
pip install -r requirements.txt
python S3_spi_middle_draa.py S1_Monthly_rainfall --out results --verify
```

The CSV gives the same results: `python S3_spi_middle_draa.py S1_monthly_precipitation_1989-2020.csv --out results --verify`.

The folder `results` then contains:

- `S1_monthly_precipitation_1989-2020_as_read.csv`: the precipitation series exactly as read by the script;
- `report.txt`: all the tables in one file, plus the verification against the values printed in the article (the last line should read `ALL VALUES MATCH THE ARTICLE`);
- `S2_SPI_monthly_series_1989-2020.csv`: the SPI series (identical to the file `S2` in this repository);
- `table1_station_statistics.csv`, `table3_table5_intensity.csv`, `table4_year_classes.csv`, `table4_frequencies.csv`, `trends_mann_kendall_sen.csv`.

To compare with the SPI Generator output, add `--compare S4_SPI_Generator_series_1989-2020.csv`.

Tested with Python 3.13, NumPy 2.5, pandas 3.0 and SciPy 1.18.

## Method in brief

- SPI: gamma distribution fitted separately to each calendar month and timescale (3, 6, 9, 12 months) by maximum likelihood (Thom approximation). Zero totals are handled with the mixed distribution H(x) = q + (1 − q)G(x). The SPI is the standard normal quantile of H. Calibration period: 1989–2020 for all stations.
- Missing values: a window containing a missing month gets no SPI and is excluded from the fit and the statistics (S1 has none).
- Drought month: SPI ≤ −1. Maximum duration: longest run of drought months. Intensity: lowest SPI. Severity: sum of negative SPI values.
- Year classes: SPI-12 of August (hydrological year September–August), 31 hydrological years labelled 1990–2020.
- Trends: Mann–Kendall test (normal approximation with tie and continuity corrections) and Sen's slope.

The full description is in the header of `S3_spi_middle_draa.py`.
