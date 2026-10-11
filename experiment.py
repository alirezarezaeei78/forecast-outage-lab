"""Causal hourly-demand stress test; no model or parameter selection on test data."""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import platform
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

DATA_URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
DATA_SHA256 = 'b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401'
ROOT = Path(__file__).resolve().parent
SEED = 42
OUTAGES = (0, 1, 3, 6)
MODEL_CONFIG = dict(loss='poisson', max_iter=150, max_leaf_nodes=15,
                    learning_rate=0.08, min_samples_leaf=30,
                    l2_regularization=1.0, early_stopping=False,
                    categorical_features=None, random_state=SEED)
FOLDS = [('2012-07-01', '2012-08-01'), ('2012-08-01', '2012-09-01'),
         ('2012-09-01', '2012-10-01')]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_counts(cache: Path) -> tuple[pd.Series, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'uci-bike-sharing.zip'
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive.write_bytes(response.read())
    data = archive.read_bytes()
    if sha(data) != DATA_SHA256:
        raise ValueError('Dataset archive differs from the recorded version; do not silently compare changed data')
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        name = next(n for n in zipped.namelist() if n.endswith('hour.csv'))
        raw = zipped.read(name)
    frame = pd.read_csv(io.BytesIO(raw))
    timestamps = pd.to_datetime(frame['dteday']) + pd.to_timedelta(frame['hr'], unit='h')
    if timestamps.duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    counts = pd.Series(frame['cnt'].to_numpy(dtype=float), index=timestamps).sort_index()
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    grid = pd.date_range(counts.index.min(), counts.index.max(), freq='h')
    counts = counts.reindex(grid)  # Missing hours remain missing, never filled with zero.
    metadata = dict(url=DATA_URL, archive_sha256=sha(data), hour_csv_sha256=sha(raw),
                    csv_rows=len(frame), hourly_grid_rows=len(grid),
                    missing_hourly_timestamps=int(counts.isna().sum()),
                    source_start=str(grid.min()), source_end=str(grid.max()),
                    license='CC BY 4.0', doi='https://doi.org/10.24432/C5W894',
                    citation='Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository.')
    return counts, metadata


def features(counts: pd.Series, outage: int = 0) -> pd.DataFrame:
    """Predict y[t] at the end of t-1; hide the most recent k hourly reports."""
    if isinstance(outage, bool) or not isinstance(outage, (int, np.integer)) or not 0 <= outage <= 23:
        raise ValueError('outage must be an integer from 0 through 23 hours')
    if not isinstance(counts.index, pd.DatetimeIndex) or not counts.index.is_unique:
        raise ValueError('Unique DatetimeIndex required')
    if len(counts) > 1 and not np.all(np.diff(counts.index.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError('Reindex onto a complete hourly grid first')
    if np.isinf(counts.to_numpy(dtype=float)).any() or (counts.dropna() < 0).any():
        raise ValueError('Counts must be nonnegative; missing counts may be NaN')
    idx = counts.index
    x = pd.DataFrame({'hour': idx.hour, 'weekday': idx.dayofweek,
                      'month': idx.month, 'year': idx.year}, index=idx, dtype=float)
    history = np.column_stack([counts.shift(i).to_numpy() for i in range(1, 25)])
    history[:, :outage] = np.nan
    for lag in (1, 2, 3):
        x[f'lag_{lag}h'] = history[:, lag - 1]
    for lag in (24, 168):
        x[f'lag_{lag}h'] = counts.shift(lag)
    available = np.isfinite(history).sum(axis=1)
    x['past24_mean'] = np.divide(np.nansum(history, axis=1), available,
                                out=np.full(len(counts), np.nan), where=available > 0)
    x['past24_available_fraction'] = available / 24.0
    return x


def mask_training(counts: pd.Series, index: pd.DatetimeIndex, seed: int) -> tuple[pd.DataFrame, dict]:
    choices = np.random.default_rng(seed).choice(OUTAGES, size=len(index),
                                                p=[0.5, 1/6, 1/6, 1/6])
    x = features(counts, 0).loc[index].copy()
    for outage in OUTAGES[1:]:
        selected = index[choices == outage]
        x.loc[selected] = features(counts, outage).loc[selected]
    return x, {str(k): int((choices == k).sum()) for k in OUTAGES}


def seasonal_fallback(x: pd.DataFrame, train_median: float) -> np.ndarray:
    """Try a week ago, then a day ago, then train-only median."""
    return x['lag_168h'].fillna(x['lag_24h']).fillna(train_median).to_numpy()


def run_checks() -> dict:
    checks = []
    idx = pd.date_range('2020-01-01', periods=220, freq='h')
    y = pd.Series(np.arange(220, dtype=float), index=idx)
    for k in OUTAGES:
        x = features(y, k)
        assert np.isnan(x.iloc[180]['lag_1h']) if k else x.iloc[180]['lag_1h'] == 179
        assert x.iloc[180]['lag_24h'] == 156 and x.iloc[180]['lag_168h'] == 12
        expected = np.arange(156, 180-k).mean()
        assert np.isclose(x.iloc[180]['past24_mean'], expected)
        assert np.isclose(x.iloc[180]['past24_available_fraction'], (24-k)/24)
        checks.append(f'{k}h masking, causal lags, rolling mean and availability')
    changed = y.copy(); changed.iloc[180:] = 999999
    pd.testing.assert_frame_equal(features(y).iloc[:181], features(changed).iloc[:181])
    checks.append('Current/future target perturbation cannot change present/past features')
    altered = y.copy(); altered.iloc[177:180] = 999999
    pd.testing.assert_series_equal(features(y, 3).iloc[180], features(altered, 3).iloc[180])
    checks.append('Hidden reports cannot affect masked features')
    missing = y.copy(); missing.iloc[10] = np.nan
    assert np.isnan(features(missing).iloc[11]['lag_1h'])
    assert features(missing).iloc[11]['past24_available_fraction'] == 10/24
    checks.append('Naturally missing hours are not converted to zero')
    masked1, counts1 = mask_training(y, idx[168:], SEED)
    masked2, counts2 = mask_training(y, idx[168:], SEED)
    pd.testing.assert_frame_equal(masked1, masked2); assert counts1 == counts2
    checks.append('Training masks deterministic for fixed seed')
    for invalid in (-1, 24, 1.5, True):
        try: features(y, invalid)
        except ValueError: pass
        else: raise AssertionError(f'Accepted invalid outage {invalid}')
    checks.append('Invalid outage arguments rejected')
    try: features(y.drop(idx[5]))
    except ValueError: pass
    else: raise AssertionError('Irregular grid accepted')
    checks.append('Irregular hourly grid rejected')
    assert np.isnan(features(y).iloc[0]['past24_mean'])
    checks.append('No history yields NaN, not an invented count')
    fallback_input = pd.DataFrame({'lag_168h':[8,np.nan,np.nan], 'lag_24h':[7,6,np.nan]})
    np.testing.assert_array_equal(seasonal_fallback(fallback_input, 5), [8,6,5])
    checks.append('Fallback precedence and train-only median')
    return {'passed': len(checks), 'checks': checks}


def plot_results(pooled: pd.DataFrame, output: Path, observed: int) -> None:
    colors = ['#52677F', '#D39332', '#087F8C', '#8B5FBF']
    names = ['ordinary_hgb', 'outage_aware_hgb', 'calendar_hgb', 'seasonal_fallback']
    labels = ['Ordinary training', 'Outage-aware training', 'Calendar only', 'Seasonal fallback']
    plt.rcParams.update({'font.size':15,'text.color':'#183346','axes.labelcolor':'#183346',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(12,10), dpi=120)
    fig.subplots_adjust(left=.12,right=.97,top=.92,bottom=.32)
    for name,label,color,marker in zip(names,labels,colors,['o','s','^','D']):
        rows = pooled[pooled.model == name].sort_values('outage_hours')
        ax.plot(rows.outage_hours, rows.mae, color=color, marker=marker, label=label,
                linewidth=2.5, markersize=8)
    ax.set_xlabel('Most recent reports unavailable (hours)')
    ax.set_ylabel('MAE (rentals/hour; lower is better)')
    ax.set_xticks(OUTAGES); ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.2)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=13)
    table_rows=[]
    for name,label in zip(names,labels):
        r=pooled[pooled.model==name].set_index('outage_hours')
        table_rows.append([label]+[f'{r.loc[k,"mae"]:.1f}' for k in OUTAGES])
    table=fig.add_axes([.12,.12,.85,.14]); table.axis('off')
    cell=table.table(cellText=table_rows,colLabels=['MAE','0h','1h','3h','6h'],
                     cellLoc='center',loc='center',colWidths=[.4,.15,.15,.15,.15])
    cell.auto_set_font_size(False);cell.set_fontsize(12);cell.scale(1,1.7)
    for (i,j),c in cell.get_celld().items():
        c.set_edgecolor('#D4E0E5'); c.set_facecolor('#ECF3F7' if i==0 else 'white')
    fig.text(.12,.06,f'UCI Bike Sharing; {observed:,} observed test hours, July–September 2012.',fontsize=12)
    fig.text(.12,.035,'Three expanding chronological folds; 1-hour horizon; simulated outages; no future weather.',fontsize=11)
    fig.text(.12,.012,'Executed scikit-learn 1.4.2; fixed settings; mask seeds 42/43/44. Data: doi.org/10.24432/C5W894.',fontsize=10)
    fig.savefig(output/'outage-mae.png'); fig.savefig(output/'outage-mae.svg'); plt.close(fig)


def run(output: Path, cache: Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    checks=run_checks()
    y,metadata=load_counts(cache)
    clean=features(y)
    predictions=[]; fold_details=[]
    first_usable=y.index[0]+pd.Timedelta(hours=168)
    for fold,(start,end) in enumerate(FOLDS,1):
        train_index=y.index[(y.index>=first_usable)&(y.index<pd.Timestamp(start))&y.notna()]
        test_index=y.index[(y.index>=pd.Timestamp(start))&(y.index<pd.Timestamp(end))&y.notna()]
        assert len(train_index) and len(test_index) and train_index.max()<test_index.min()
        xtrain=clean.loc[train_index]; ytrain=y.loc[train_index]
        augmented,mask_counts=mask_training(y,train_index,SEED+fold-1)
        models={'ordinary_hgb':HistGradientBoostingRegressor(**MODEL_CONFIG),
                'outage_aware_hgb':HistGradientBoostingRegressor(**MODEL_CONFIG),
                'calendar_hgb':HistGradientBoostingRegressor(**MODEL_CONFIG)}
        with threadpool_limits(limits=2):
            models['ordinary_hgb'].fit(xtrain,ytrain)
            models['outage_aware_hgb'].fit(augmented,ytrain)
            models['calendar_hgb'].fit(xtrain[['hour','weekday','month','year']],ytrain)
            for outage in OUTAGES:
                xtest=features(y,outage).loc[test_index]
                for name,model in models.items():
                    used=xtest[['hour','weekday','month','year']] if name=='calendar_hgb' else xtest
                    pred=model.predict(used)
                    assert np.isfinite(pred).all() and (pred>=0).all()
                    predictions.append(pd.DataFrame(dict(timestamp=test_index,fold=fold,
                        outage_hours=outage,model=name,actual=y.loc[test_index].to_numpy(),predicted=pred)))
                predictions.append(pd.DataFrame(dict(timestamp=test_index,fold=fold,outage_hours=outage,
                    model='seasonal_fallback',actual=y.loc[test_index].to_numpy(),
                    predicted=seasonal_fallback(xtest,float(ytrain.median())))))
        fold_details.append(dict(fold=fold,train_start=str(train_index.min()),train_end=str(train_index.max()),
            train_rows=len(train_index),test_start=start,test_end_exclusive=end,test_rows=len(test_index),
            masking_seed=SEED+fold-1,training_outage_counts=mask_counts,trees_per_model=150))
        print(f'Completed fold {fold}: train={len(train_index)}, test={len(test_index)}',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    assert not pred.duplicated(['timestamp','outage_hours','model']).any()
    assert pred.groupby(['outage_hours','model']).size().nunique()==1
    records=[]
    for (fold,outage,model),frame in pred.groupby(['fold','outage_hours','model']):
        records.append(dict(fold=int(fold),outage_hours=int(outage),model=model,n=len(frame),
            mae=float(mean_absolute_error(frame.actual,frame.predicted)),
            rmse=float(np.sqrt(mean_squared_error(frame.actual,frame.predicted)))))
    fold_scores=pd.DataFrame(records); fold_scores.to_csv(output/'fold-metrics.csv',index=False)
    records=[]
    for (outage,model),frame in pred.groupby(['outage_hours','model']):
        records.append(dict(outage_hours=int(outage),model=model,n=len(frame),
            mae=float(mean_absolute_error(frame.actual,frame.predicted)),
            rmse=float(np.sqrt(mean_squared_error(frame.actual,frame.predicted)))))
    pooled=pd.DataFrame(records);pooled.to_csv(output/'metrics.csv',index=False)
    pred.to_csv(output/'predictions.csv',index=False)
    plot_results(pooled,output,int(pooled.n.iloc[0]))
    checks['checks'] += ['Strict chronological train/test ordering in every fold',
                        'Finite nonnegative forecasts for all models/scenarios',
                        'Identical target rows for all model/scenario comparisons',
                        'Nonoverlapping test timestamps and unique predictions']
    checks['passed']=len(checks['checks'])
    result=dict(executed_at=datetime.now(timezone.utc).isoformat(),data=metadata,folds=fold_details,
        horizon_hours=1,outages=list(OUTAGES),model_config=MODEL_CONFIG,
        training_mask_probabilities={'0':.5,'1':1/6,'3':1/6,'6':1/6},
        no_test_tuning=True,refit_at_month_start=True,weather_features_used=False,
        target_definition='cnt (observed hourly total rentals)',
        outage_definition='At EACH rolling forecast origin hide the preceding k hourly reports; older reports recovered. Not one unrecovered quarter-long outage.',
        environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                     'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,
                     'threadpoolctl':importlib.metadata.version('threadpoolctl'),
                     'platform':platform.platform(),'processor':platform.processor(),'threads':2},
        command='python experiment.py --output results',checks=checks,
        pooled_metrics=pooled.to_dict(orient='records'),
        limitations=['One historical city and one quarter; not a production deployment or SOTA benchmark.',
                     'One fixed masking draw per fold (seeds 42/43/44); no inferential confidence intervals.',
                     'Features may use earlier TEST-period counts only when available at that rolling forecast origin.',
                     'No models/parameters selected on held-out results; first 168 hours excluded from training.',
                     'Original missing targets are not evaluated. Missing past reports remain NaN.',
                     '50% row-wise training masking is a preset stress-test recipe, not an estimated real outage distribution.',
                     '0h means no ADDED outage; natural gaps remain. Outages do not extend to seasonal lags (24/168h).'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(pooled.to_string(index=False));print(f'Checks passed: {checks["passed"]}',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--cache',type=Path,default=ROOT/'data')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only: print(json.dumps(run_checks(),indent=2))
    else: run(args.output,args.cache)
