"""
Supplementary Material S3 (version 1.1)
Reproducible SPI calculation and drought statistics for the Middle Draa basin, 1989-2020.

Article: "Multi-scale analysis of meteorological drought: a case study in the Middle Draa Valley,
southeastern Morocco" (Ecological Engineering & Environmental Technology).
Repository: https://github.com/mohamedmanaouch/spi_materials

------------------------------------------------------------------------------------------------
QUICK START
    pip install numpy pandas scipy openpyxl
    python S3_spi_middle_draa.py S1_Monthly_rainfall --out results --verify
    (or, equivalently: python S3_spi_middle_draa.py S1_monthly_precipitation_1989-2020.csv --out results --verify)

This writes every table of the article into the folder "results" and, with --verify, checks that
the values match those printed in the article.

INPUT (S1), either:
    - the folder S1_Monthly_rainfall with the four station workbooks (Station Agdez.xlsx, Station Zagora.xlsx,
      Station Tagounite.xlsx, Station M'hamid El ghizlane.xlsx: year, month, monthly total in mm), or
    - a CSV file, one row per month, first column "date" (YYYY-MM), then one column per station.
    Empty cells are treated as missing values. Months outside the analysis period (e.g. Agdez 2021-2022) are ignored.
    The series actually used is written to results/S1_monthly_precipitation_1989-2020_as_read.csv.

ALGORITHM (McKee et al., 1993; Edwards & McKee, 1997)
    For each station, timescale k (3, 6, 9, 12 months) and calendar month j:
    1. x = sum of the monthly totals over the k months ending in month j (NaN if any month is missing).
    2. Calibration sample = values of x for the years of the calibration period (default 1989-2020).
       q = share of zeros in the calibration sample.
    3. Two-parameter gamma distribution fitted to the non-zero calibration values by maximum likelihood,
       Thom (1958) approximation:  A = ln(mean) - mean(ln x),  alpha = (1 + sqrt(1 + 4A/3)) / (4A),
       beta = mean / alpha.
    4. H(x) = q + (1 - q) * G(x; alpha, beta)   (G = gamma CDF; x = 0 gives H = q).
       H is bounded to [1e-6, 1 - 1e-6] to avoid infinite values.
    5. SPI = PHI^-1(H)   (PHI^-1 = inverse standard normal CDF).
    Missing months: a k-month window containing a missing month gets no SPI value; it is left out of the
    fit and of all statistics. (The four series used in the article have no missing month.)

DROUGHT STATISTICS (period 1989-2020)
    drought month      SPI <= -1
    maximum duration   longest run of consecutive drought months
    intensity          lowest SPI value
    severity           sum of all negative SPI values
    year class         SPI-12 of August (hydrological year September-August), classes of McKee et al. (1993):
                       >= 2 extreme wetness; [1.5, 2) high wetness; [0, 1.5) near normal to moderately wet;
                       (-1.5, 0) near normal to moderately dry; (-2, -1.5] severe drought; <= -2 extreme drought.
                       Hydrological years 1989/90 to 2019/20 (31 years, labelled 1990-2020).
    trend              Mann-Kendall test (normal approximation with tie correction, two-sided) and
                       Sen's slope, on the August SPI-12 series and on hydrological-year precipitation.

OPTIONS
    --calib Y1 Y2    calibration years (default 1989 2020)
    --period Y1 Y2   analysis period (default 1989 2020)
    --compare FILE   compare the SPI with another SPI file (same column names, e.g. SPI Generator output)
    --verify         check the results against the values published in the article
------------------------------------------------------------------------------------------------
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

VERSION = '1.1'
SCALES = (3, 6, 9, 12)
CLASSES = ['Extreme wetness', 'High wetness', 'Near normal to moderately wet', 'Near normal to moderately dry',
           'Severe drought', 'Extreme drought']


# ----------------------------------------------------------------------------------------------- SPI
def gamma_thom(x):
    """Thom (1958) maximum-likelihood approximation of gamma shape (alpha) and scale (beta), non-zero values."""
    x = np.asarray(x, float)
    x = x[x > 0]
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan, np.nan
    mean = x.mean()
    a = np.log(mean) - np.log(x).mean()
    alpha = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    return alpha, mean / alpha


def spi(precip, k, calib=(1989, 2020)):
    """SPI-k of a monthly precipitation series (pd.Series with a monthly PeriodIndex)."""
    acc = precip.rolling(k, min_periods=k).sum()       # NaN if the window contains a missing month
    out = pd.Series(np.nan, index=precip.index)
    for m in range(1, 13):
        sel = acc[acc.index.month == m].dropna()
        ref = sel[(sel.index.year >= calib[0]) & (sel.index.year <= calib[1])]
        if len(ref) < 2:
            continue
        q = float((ref == 0).mean())
        alpha, beta = gamma_thom(ref.values)
        if np.isnan(alpha):
            continue
        h = q + (1 - q) * stats.gamma.cdf(sel.values, a=alpha, scale=beta)
        out[sel.index] = stats.norm.ppf(np.clip(h, 1e-6, 1 - 1e-6))
    return out


# ----------------------------------------------------------------------------------------------- statistics
def max_run(s, thr=-1.0):
    run = best = 0
    for v in s.values:
        run = run + 1 if (not np.isnan(v) and v <= thr) else 0
        best = max(best, run)
    return best


def classify(v):
    if v >= 2:
        return CLASSES[0]
    if v >= 1.5:
        return CLASSES[1]
    if v >= 0:
        return CLASSES[2]
    if v > -1.5:
        return CLASSES[3]
    if v > -2:
        return CLASSES[4]
    return CLASSES[5]


def mann_kendall(y):
    """Mann-Kendall test (normal approximation, tie correction, continuity correction) and Sen's slope."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(c * (c - 1) * (2 * c + 5) for c in counts)) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return {'n': n, 'S': int(s), 'Z': round(float(z), 2), 'p_value': round(float(p), 3),
            'sen_slope_per_year': round(float(np.median(slopes)), 4)}


def hydro_year_totals(p, y0, y1):
    hy = p.copy()
    hy.index = [(d.year + 1 if d.month >= 9 else d.year) for d in hy.index]
    tot = hy.groupby(level=0).sum(min_count=12)
    return tot.loc[y0 + 1:y1]


# ----------------------------------------------------------------------------------------------- input
STATION_FILES = {'Agdez': 'Station Agdez.xlsx', 'Zagora': 'Station Zagora.xlsx',
                 'Tagounite': 'Station Tagounite.xlsx', 'Mhamid_El_Ghizlane': "Station M'hamid El ghizlane.xlsx"}


def read_station_xlsx(path):
    """Read one station workbook (year, month and precipitation columns, in any position and any sheet)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True)
                if sum(isinstance(v, (int, float)) for v in r if v is not None) >= 3]
        if len(rows) < 12:
            continue
        ncol = max(len(r) for r in rows)
        cols = [[r[i] if i < len(r) else None for r in rows] for i in range(ncol)]
        num = [i for i, c in enumerate(cols) if all(isinstance(v, (int, float)) for v in c)]
        year = [i for i in num if all(1900 <= v <= 2100 for v in cols[i])]
        month = [i for i in num if set(cols[i]) <= set(range(1, 13)) and len(set(cols[i])) == 12]
        value = [i for i in num if i not in year + month]
        if len(year) == 1 and len(month) == 1 and len(value) == 1:
            idx = [f'{int(y)}-{int(m):02d}' for y, m in zip(cols[year[0]], cols[month[0]])]
            return pd.Series([float(v) for v in cols[value[0]]], index=idx)
    raise ValueError(f'no year/month/value table found in {path}')


def read_input(path):
    """CSV file (date, one column per station) or folder with the four station workbooks (S1_Monthly_rainfall)."""
    if os.path.isdir(path):
        P = pd.DataFrame({st: read_station_xlsx(os.path.join(path, f)) for st, f in STATION_FILES.items()})
        P.index.name = 'date'
        P = P.reset_index()
    else:
        P = pd.read_csv(path)
    P.index = pd.PeriodIndex(P.pop('date'), freq='M')
    return P.astype(float).sort_index()


# ----------------------------------------------------------------------------------------------- values printed in the article
PUBLISHED = {
    # values printed in the revised article (Tables 1, 3, 4, 5, intensities, Mann-Kendall and Sen)
    ('t1', 'Agdez', 'MAP_mm'): 93.5, ('t1', 'Agdez', 'SD_mm'): 45.72,
    ('t1', 'Zagora', 'MAP_mm'): 67.25, ('t1', 'Zagora', 'SD_mm'): 30.05,
    ('t1', 'Tagounite', 'MAP_mm'): 52.03, ('t1', 'Tagounite', 'SD_mm'): 32.32,
    ('t1', 'Mhamid_El_Ghizlane', 'MAP_mm'): 39.16, ('t1', 'Mhamid_El_Ghizlane', 'SD_mm'): 28.28,
    ('ch', ('Agdez', 3), 'intensity_min_SPI'): -1.86, ('ch', ('Agdez', 3), 'severity_sum_negative_SPI'): -140.22,
    ('ch', ('Agdez', 6), 'intensity_min_SPI'): -2.34, ('ch', ('Agdez', 6), 'severity_sum_negative_SPI'): -153.41,
    ('ch', ('Agdez', 9), 'intensity_min_SPI'): -2.41, ('ch', ('Agdez', 9), 'severity_sum_negative_SPI'): -151.92,
    ('ch', ('Agdez', 12), 'intensity_min_SPI'): -2.46, ('ch', ('Agdez', 12), 'severity_sum_negative_SPI'): -152.11,
    ('ch', ('Agdez', 12), 'max_duration_months'): 18,
    ('ch', ('Zagora', 3), 'intensity_min_SPI'): -1.86, ('ch', ('Zagora', 3), 'severity_sum_negative_SPI'): -139.7,
    ('ch', ('Zagora', 6), 'intensity_min_SPI'): -2.63, ('ch', ('Zagora', 6), 'severity_sum_negative_SPI'): -151.48,
    ('ch', ('Zagora', 9), 'intensity_min_SPI'): -2.96, ('ch', ('Zagora', 9), 'severity_sum_negative_SPI'): -154.68,
    ('ch', ('Zagora', 12), 'intensity_min_SPI'): -2.26, ('ch', ('Zagora', 12), 'severity_sum_negative_SPI'): -154.89,
    ('ch', ('Zagora', 12), 'max_duration_months'): 17,
    ('ch', ('Tagounite', 3), 'intensity_min_SPI'): -1.32, ('ch', ('Tagounite', 3), 'severity_sum_negative_SPI'): -93.35,
    ('ch', ('Tagounite', 6), 'intensity_min_SPI'): -2.37, ('ch', ('Tagounite', 6), 'severity_sum_negative_SPI'): -141.36,
    ('ch', ('Tagounite', 9), 'intensity_min_SPI'): -2.72, ('ch', ('Tagounite', 9), 'severity_sum_negative_SPI'): -148.16,
    ('ch', ('Tagounite', 12), 'intensity_min_SPI'): -2.3, ('ch', ('Tagounite', 12), 'severity_sum_negative_SPI'): -148.97,
    ('ch', ('Tagounite', 12), 'max_duration_months'): 11,
    ('ch', ('Mhamid_El_Ghizlane', 3), 'intensity_min_SPI'): -0.78, ('ch', ('Mhamid_El_Ghizlane', 3), 'severity_sum_negative_SPI'): -48.55,
    ('ch', ('Mhamid_El_Ghizlane', 6), 'intensity_min_SPI'): -1.85, ('ch', ('Mhamid_El_Ghizlane', 6), 'severity_sum_negative_SPI'): -116.57,
    ('ch', ('Mhamid_El_Ghizlane', 9), 'intensity_min_SPI'): -2.13, ('ch', ('Mhamid_El_Ghizlane', 9), 'severity_sum_negative_SPI'): -144.76,
    ('ch', ('Mhamid_El_Ghizlane', 12), 'intensity_min_SPI'): -2.62, ('ch', ('Mhamid_El_Ghizlane', 12), 'severity_sum_negative_SPI'): -145.68,
    ('ch', ('Mhamid_El_Ghizlane', 12), 'max_duration_months'): 13,
    ('cnt', 'Agdez', 'Extreme wetness'): 1,
    ('cnt', 'Agdez', 'High wetness'): 3,
    ('cnt', 'Agdez', 'Near normal to moderately wet'): 7,
    ('cnt', 'Agdez', 'Near normal to moderately dry'): 18,
    ('cnt', 'Agdez', 'Severe drought'): 1,
    ('cnt', 'Agdez', 'Extreme drought'): 1,
    ('cnt', 'Zagora', 'Extreme wetness'): 1,
    ('cnt', 'Zagora', 'High wetness'): 1,
    ('cnt', 'Zagora', 'Near normal to moderately wet'): 11,
    ('cnt', 'Zagora', 'Near normal to moderately dry'): 16,
    ('cnt', 'Zagora', 'Severe drought'): 2,
    ('cnt', 'Zagora', 'Extreme drought'): 0,
    ('cnt', 'Tagounite', 'Extreme wetness'): 0,
    ('cnt', 'Tagounite', 'High wetness'): 2,
    ('cnt', 'Tagounite', 'Near normal to moderately wet'): 13,
    ('cnt', 'Tagounite', 'Near normal to moderately dry'): 14,
    ('cnt', 'Tagounite', 'Severe drought'): 1,
    ('cnt', 'Tagounite', 'Extreme drought'): 1,
    ('cnt', 'Mhamid_El_Ghizlane', 'Extreme wetness'): 1,
    ('cnt', 'Mhamid_El_Ghizlane', 'High wetness'): 1,
    ('cnt', 'Mhamid_El_Ghizlane', 'Near normal to moderately wet'): 15,
    ('cnt', 'Mhamid_El_Ghizlane', 'Near normal to moderately dry'): 12,
    ('cnt', 'Mhamid_El_Ghizlane', 'Severe drought'): 1,
    ('cnt', 'Mhamid_El_Ghizlane', 'Extreme drought'): 1,
    ('avg', None, 'Extreme wetness'): 2.42,
    ('avg', None, 'High wetness'): 5.65,
    ('avg', None, 'Near normal to moderately wet'): 37.1,
    ('avg', None, 'Near normal to moderately dry'): 48.39,
    ('avg', None, 'Severe drought'): 4.03,
    ('avg', None, 'Extreme drought'): 2.42,
    ('mk', 'Agdez', 'Z'): -0.94, ('mk', 'Agdez', 'p_value'): 0.35, ('mk', 'Agdez', 'sen_slope_per_year'): -0.0115,
    ('mk', 'Zagora', 'Z'): -1.7, ('mk', 'Zagora', 'p_value'): 0.089, ('mk', 'Zagora', 'sen_slope_per_year'): -0.0263,
    ('mk', 'Tagounite', 'Z'): -1.92, ('mk', 'Tagounite', 'p_value'): 0.055, ('mk', 'Tagounite', 'sen_slope_per_year'): -0.0407,
    ('mk', 'Mhamid_El_Ghizlane', 'Z'): -0.99, ('mk', 'Mhamid_El_Ghizlane', 'p_value'): 0.323, ('mk', 'Mhamid_El_Ghizlane', 'sen_slope_per_year'): -0.0226,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('precip_csv')
    ap.add_argument('--out', default='results')
    ap.add_argument('--calib', nargs=2, type=int, default=[1989, 2020])
    ap.add_argument('--period', nargs=2, type=int, default=[1989, 2020])
    ap.add_argument('--compare')
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    y0, y1 = a.period
    lines = [f'S3_spi_middle_draa.py version {VERSION} | Python {sys.version.split()[0]} | numpy {np.__version__} | '
             f'pandas {pd.__version__} | scipy {__import__("scipy").__version__}',
             f'Input: {a.precip_csv} | calibration {a.calib[0]}-{a.calib[1]} | analysis period {y0}-{y1}', '']

    P = read_input(a.precip_csv)
    P = P[(P.index.year >= y0) & (P.index.year <= y1)]
    P = P.reindex(pd.period_range(f'{y0}-01', f'{y1}-12', freq='M'))
    P.round(3).to_csv(f'{a.out}/S1_monthly_precipitation_{y0}-{y1}_as_read.csv', index_label='date')

    # ---- input check and Table 1
    t1 = []
    for st in P.columns:
        s = P[st][(P.index.year >= y0) & (P.index.year <= y1)]
        ann = s.groupby(s.index.year).sum(min_count=12)
        t1.append({'station': st, 'months': int(s.notna().sum()), 'missing_months': int(s.isna().sum()),
                   'years': int(ann.notna().sum()), 'share_zero_months_%': round(100 * float((s == 0).mean()), 1),
                   'MAP_mm': round(float(ann.mean()), 2), 'SD_mm': round(float(ann.std(ddof=1)), 2)})
    t1 = pd.DataFrame(t1)
    t1.to_csv(f'{a.out}/table1_station_statistics.csv', index=False)

    # ---- SPI (S2)
    cols = {}
    for st in P.columns:
        for k in SCALES:
            cols[f'{st}_SPI{k}'] = spi(P[st], k, tuple(a.calib))
    S = pd.DataFrame(cols)
    S = S[(S.index.year >= y0) & (S.index.year <= y1)]
    S.round(3).to_csv(f'{a.out}/S2_SPI_monthly_series_{y0}-{y1}.csv', index_label='date', na_rep='')

    # ---- Tables 3 and 5, intensity
    rows = []
    for st in P.columns:
        for k in SCALES:
            s = S[f'{st}_SPI{k}'].dropna()
            rows.append({'station': st, 'scale': f'SPI-{k}', 'n_values': len(s), 'max_duration_months': max_run(s),
                         'intensity_min_SPI': round(float(s.min()), 2),
                         'severity_sum_negative_SPI': round(float(s[s < 0].sum()), 2),
                         'mean_SPI': round(float(s.mean()), 2)})
    ch = pd.DataFrame(rows)
    ch.to_csv(f'{a.out}/table3_table5_intensity.csv', index=False)

    # ---- Table 4
    yrs = []
    for st in P.columns:
        aug = S[f'{st}_SPI12'][(S.index.month == 8) & (S.index.year >= y0 + 1)]
        for d, v in aug.items():
            yrs.append({'station': st, 'hydro_year': d.year, 'SPI12_August': round(float(v), 2), 'class': classify(v)})
    yd = pd.DataFrame(yrs)
    yd.to_csv(f'{a.out}/table4_year_classes.csv', index=False)
    cnt = pd.crosstab(yd['station'], yd['class']).reindex(columns=CLASSES, fill_value=0).reindex(P.columns)
    pct = (100 * cnt.div(cnt.sum(axis=1), axis=0)).round(2)
    avg = (100 * cnt.sum() / cnt.values.sum()).round(2)
    freq = pd.concat({'count': cnt, 'percent': pct}, axis=1)
    freq.to_csv(f'{a.out}/table4_frequencies.csv')

    # ---- trends
    tr = []
    for st in P.columns:
        aug = yd[yd.station == st].sort_values('hydro_year')
        aug_full = S[f'{st}_SPI12'][(S.index.month == 8) & (S.index.year >= y0 + 1)]
        tr.append({'station': st, 'series': 'August SPI-12', **mann_kendall(aug_full.values)})
        tr.append({'station': st, 'series': 'Hydrological-year precipitation (mm)',
                   **mann_kendall(hydro_year_totals(P[st], y0, y1).values)})
    tr = pd.DataFrame(tr)
    tr.to_csv(f'{a.out}/trends_mann_kendall_sen.csv', index=False)

    # ---- report
    pd.set_option('display.width', 200)
    lines += ['TABLE 1. Station statistics (calendar years)', t1.to_string(index=False), '',
              'TABLE 3 / INTENSITY / TABLE 5. Drought characteristics', ch.to_string(index=False), '',
              'TABLE 4. Classes of hydrological years (August SPI-12): counts and %', freq.to_string(), '',
              'Average over the stations (%):', avg.to_string(), '',
              'Years per class:']
    for st in P.columns:
        lines.append(f'  {st}')
        for c in CLASSES:
            ys = yd[(yd.station == st) & (yd['class'] == c)].hydro_year.tolist()
            lines.append(f'    {c}: {", ".join(map(str, ys)) or "-"}')
    lines += ['', 'TRENDS. Mann-Kendall test and Sen slope', tr.to_string(index=False), '']

    # ---- optional comparison with another SPI file (e.g. SPI Generator output)
    if a.compare:
        R = pd.read_csv(a.compare)
        R.index = pd.PeriodIndex(R.pop('date'), freq='M')
        cmp_rows = []
        for c in S.columns:
            if c in R:
                j = pd.concat([S[c], R[c]], axis=1).dropna()
                d = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
                cmp_rows.append({'series': c, 'n': len(j), 'r': round(float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]), 4),
                                 'mean_abs_diff': round(float(d.mean()), 4), 'median_abs_diff': round(float(d.median()), 4),
                                 'n_diff_gt_0.05': int((d > 0.05).sum())})
        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(f'{a.out}/comparison_with_{os.path.splitext(os.path.basename(a.compare))[0]}.csv', index=False)
        lines += [f'COMPARISON WITH {a.compare}', cmp_df.to_string(index=False), '']

    # ---- verification against the article
    if a.verify:
        ok = True
        def check(label, got, exp, tol):
            nonlocal ok
            good = abs(got - exp) <= tol
            ok &= good
            lines.append(f'  [{"OK" if good else "MISMATCH"}] {label}: script {got} | article {exp}')
        lines.append('VERIFICATION AGAINST THE ARTICLE')
        for (kind, st, key), exp in PUBLISHED.items():
            if kind == 't1':
                got = float(t1.set_index('station').loc[st, key]); tol = 0.005
            elif kind == 'ch':
                stn, k = st
                got = float(ch[(ch.station == stn) & (ch.scale == f'SPI-{k}')][key].iloc[0]); tol = 0.005
            elif kind == 'cnt':
                got = float(cnt.loc[st, key]); tol = 0
            elif kind == 'avg':
                got = float(avg[key]); tol = 0.005
            elif kind == 'mk':
                got = float(tr[(tr.station == st) & (tr.series == 'August SPI-12')][key].iloc[0]); tol = 0.005
            check(f'{kind} {st} {key}', got, exp, tol)
        lines.append('ALL VALUES MATCH THE ARTICLE' if ok else 'SOME VALUES DIFFER FROM THE ARTICLE')

    rep = '\n'.join(lines)
    open(f'{a.out}/report.txt', 'w', encoding='utf-8').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
