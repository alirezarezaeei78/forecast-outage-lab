"""Reusable freshness gate and exploratory replay of the frozen October 11 forecasts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from experiment import ROOT, OUTAGES, features, load_counts, run_checks as feature_checks
import matplotlib.pyplot as plt

BASE_SHA='0695d32b6d75231478cdb849c4305490af836fb989db3ea3ddb8817f13ab493b'
BASE_COMMIT='4f7902fa147b5f8c40553d14363e45254dc54cce'
SEEDS=(101,102,103,104,105)
MODELS=('ordinary_hgb','outage_aware_hgb','calendar_hgb','seasonal_fallback','age_gate')
LABELS=('Always ordinary','Always outage-aware','Always calendar','Seasonal fallback','Age gate')

def vector(value, name):
    a=np.asarray(value,dtype=float)
    if a.ndim!=1 or not len(a): raise ValueError(f'{name} must be a nonempty one-dimensional array')
    return a

def route_forecasts(age_hours, ordinary, fallback):
    """Use ordinary only at age zero with a valid prediction; otherwise use fallback.

    Age is delay relative to the expected report at t-1, not age relative to t.
    Unknown age (NaN/+inf) falls back. Does not read outcomes or errors.
    """
    age,primary,backup=[vector(v,n) for v,n in zip((age_hours,ordinary,fallback),('age','ordinary','fallback'))]
    if not (age.shape==primary.shape==backup.shape): raise ValueError('Array shapes must match')
    if (age<0).any() or ((np.isfinite(age)) & (age!=np.floor(age))).any(): raise ValueError('Age must be whole nonnegative hours or unknown')
    if not np.isfinite(backup).all() or (backup<0).any(): raise ValueError('Fallback must be finite and nonnegative')
    use=(age==0)&np.isfinite(primary)&(primary>=0)
    return np.where(use,primary,backup),~use

def report_age_hours(counts: pd.Series, index: pd.DatetimeIndex, delay):
    """Find most recent available report in past 24h; missing history => inf."""
    features(counts,0)  # Shared input/grid validation; no target-derived routing.
    if not index.is_unique or not index.isin(counts.index).all(): raise ValueError('Unique forecast times must be on the hourly grid')
    d=np.asarray(delay)
    if d.ndim==0: d=np.repeat(d,len(index))
    if d.shape!=(len(index),) or d.dtype.kind not in 'iu' or (d<0).any() or (d>23).any():
        raise ValueError('Delay must be integer 0..23, one per timestamp')
    available=np.column_stack([counts.shift(lag).loc[index].notna().to_numpy() & (d<lag) for lag in range(1,25)])
    return np.where(available.any(axis=1),available.argmax(axis=1).astype(float),np.inf)

def mixed_delays(index, seed):
    days=pd.DatetimeIndex(index.normalize().unique()).sort_values()
    choices=np.random.default_rng(seed).choice(OUTAGES,len(days),p=[.7,.1,.1,.1])
    return pd.Series(choices,index=days).reindex(index.normalize()).to_numpy()

def run_checks():
    checks=feature_checks()['checks'].copy()
    idx=pd.date_range('2020-01-01',periods=60,freq='h'); y=pd.Series(np.arange(60,dtype=float),index=idx)
    for k in OUTAGES: assert report_age_hours(y,idx[30:31],k)[0]==k
    checks.append('Report-age convention equals declared delay when history is complete')
    missing=y.copy(); missing.iloc[29]=np.nan
    assert report_age_hours(missing,idx[30:31],0)[0]==1
    assert report_age_hours(y,idx[:1],0)[0]==np.inf
    checks.append('Natural missing report and no history trigger stale/unknown ages')
    changed=y.copy(); changed.iloc[30:]=np.nan
    np.testing.assert_array_equal(report_age_hours(y,idx[:31],0),report_age_hours(changed,idx[:31],0))
    checks.append('Changing current/future target availability cannot alter present/past ages')
    hidden=y.copy(); hidden.iloc[27:30]=np.nan
    np.testing.assert_array_equal(report_age_hours(y,idx[30:31],3),report_age_hours(hidden,idx[30:31],3))
    checks.append('Hidden observations cannot change masked report age')
    pred,used=route_forecasts([0,1,np.nan,np.inf,0,0],[1,2,3,4,np.nan,-1],[7]*6)
    np.testing.assert_array_equal(pred,[1,7,7,7,7,7]); np.testing.assert_array_equal(used,[False,True,True,True,True,True])
    checks.append('Fresh, stale, unknown and invalid-primary routing preserves fallback values')
    args=([0,1],[2,3],[8,9]); copies=[np.array(a).copy() for a in args]; route_forecasts(*copies)
    for original,copied in zip(args,copies): np.testing.assert_array_equal(original,copied)
    checks.append('Routing leaves caller arrays unchanged')
    for age,primary,backup in [([],[],[]),([0],[1,2],[3]),([[0]],[1],[2]),([-1],[1],[2]),([.5],[1],[2]),([0],[1],[np.nan]),([0],[1],[-1]),([0],[1],[np.inf])]:
        try: route_forecasts(age,primary,backup)
        except ValueError: pass
        else: raise AssertionError('Accepted malformed routing input')
    checks.append('Bad shapes, ages and fallback numbers rejected')
    for d in (True,1.5,-1,24,[0,1]):
        try: report_age_hours(y,idx[30:31],d)
        except ValueError: pass
        else: raise AssertionError('Accepted invalid reporting delay')
    checks.append('Invalid delay arguments rejected')
    for seed in SEEDS:
        d=mixed_delays(idx,seed); np.testing.assert_array_equal(d,mixed_delays(idx,seed))
        assert pd.Series(d,index=idx).groupby(idx.normalize()).nunique().max()==1
    checks.append('All five schedules deterministic with one delay per day')
    return {'passed':len(checks),'checks':checks}

def scores(frame, keys):
    records=[]
    for values,g in frame.groupby(keys):
        if not isinstance(values,tuple): values=(values,)
        records.append(dict(zip(keys,values),n=len(g),mae=float(mean_absolute_error(g.actual,g.predicted)),
            rmse=float(np.sqrt(mean_squared_error(g.actual,g.predicted))),fallback_fraction=float(g.fallback_used.mean())))
    return pd.DataFrame(records)

def make_figure(summary, output):
    plt.rcParams.update({'font.size':15,'text.color':'#183346','axes.labelcolor':'#183346',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(12,9),dpi=120);fig.subplots_adjust(left=.3,right=.96,top=.93,bottom=.26)
    rows=summary.set_index('model').loc[list(MODELS)]
    values=rows.mae.to_numpy(); colors=['#52677F','#D39332','#8B5FBF','#697D70','#087F8C']
    ax.barh(range(5),values,color=colors);ax.set_yticks(range(5),LABELS);ax.invert_yaxis()
    ax.set_xlim(0,max(values)*1.16);ax.set_xlabel('MAE (rentals/hour; lower is better)');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for i,v in enumerate(values): ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=16)
    gate=rows.loc['age_gate']
    fig.text(.05,.18,f'Age gate: ordinary if report age = 0; otherwise calendar. Fallback: {100*gate.fallback_fraction:.1f}% of origins.',fontsize=13)
    fig.text(.05,.13,'UCI Bike Sharing; 2,208 target hours; July–September 2012; 1-hour forecasts.',fontsize=12)
    fig.text(.05,.09,'Mean of five daily-delay replays, seeds 101–105; nominal delay probability 30%.',fontsize=12)
    fig.text(.05,.05,'Exploratory reuse of frozen October 11 forecasts; no new model fits or independent validation.',fontsize=11)
    fig.text(.05,.015,'Data: doi.org/10.24432/C5W894. Public aggregate counts, CC BY 4.0. No confidence intervals.',fontsize=10)
    fig.savefig(output/'mixed-mae.png');fig.savefig(output/'mixed-mae.svg');plt.close(fig)

def run(output, cache, source):
    output.mkdir(parents=True,exist_ok=True);checks=run_checks();raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASE_SHA: raise ValueError('Original predictions changed; replay requires the frozen public input')
    baseline=pd.read_csv(source,parse_dates=['timestamp'])
    assert not baseline.duplicated(['timestamp','outage_hours','model']).any()
    assert set(baseline.model)==set(MODELS[:-1]) and set(baseline.outage_hours)==set(OUTAGES)
    assert baseline.groupby('timestamp').actual.nunique().max()==1
    wide=baseline.pivot(index=['timestamp','fold','outage_hours','actual'],columns='model',values='predicted').reset_index()
    assert np.isfinite(wide[list(MODELS[:-1])].to_numpy()).all()
    y,metadata=load_counts(cache);times=pd.DatetimeIndex(sorted(wide.timestamp.unique()))
    assert len(times)==2208
    assert (wide.actual.to_numpy()==y.loc[wide.timestamp].to_numpy()).all()
    ages={k:report_age_hours(y,times,k) for k in OUTAGES}; frames=[]
    regimes=[(f'{k}h',0,np.repeat(k,len(times))) for k in OUTAGES]+[('mixed',seed,mixed_delays(times,seed)) for seed in SEEDS]
    for regime,seed,delays in regimes:
        choice=pd.DataFrame({'timestamp':times,'outage_hours':delays})
        rows=choice.merge(wide,on=['timestamp','outage_hours'],validate='one_to_one').sort_values('timestamp')
        age=report_age_hours(y,pd.DatetimeIndex(rows.timestamp),rows.outage_hours.to_numpy())
        gate,used=route_forecasts(age,rows.ordinary_hgb,rows.calendar_hgb)
        again,again_used=route_forecasts(age,rows.ordinary_hgb,rows.calendar_hgb)
        np.testing.assert_array_equal(gate,again);np.testing.assert_array_equal(used,again_used)
        for model in MODELS:
            frames.append(pd.DataFrame({'timestamp':rows.timestamp,'fold':rows.fold,'regime':regime,'seed':seed,
                'delay_hours':rows.outage_hours,'age_hours':age,'actual':rows.actual,'model':model,
                'predicted':gate if model=='age_gate' else rows[model],
                'fallback_used':used if model=='age_gate' else np.zeros(len(rows),dtype=bool)}))
    replay=pd.concat(frames,ignore_index=True)
    assert replay.groupby(['regime','seed','model']).size().eq(2208).all()
    assert not replay.duplicated(['timestamp','regime','seed','model']).any()
    assert np.isfinite(replay.predicted).all() and (replay.predicted>=0).all()
    checks['checks']+=['Frozen public prediction hash and UCI target identity verified',
        'Identical 2,208 rows in every schedule/method; no duplicated predictions',
        'Finite nonnegative outputs; exact repeated routing parity']
    checks['passed']=len(checks['checks'])
    metrics=scores(replay,['regime','seed','model']);folds=scores(replay,['regime','seed','fold','model'])
    summary=metrics[metrics.regime=='mixed'].groupby('model',as_index=False)[['mae','rmse','fallback_fraction']].mean()
    replay.to_csv(output/'predictions.csv',index=False);metrics.to_csv(output/'metrics.csv',index=False)
    folds.to_csv(output/'fold-metrics.csv',index=False);summary.to_csv(output/'mixed-summary.csv',index=False)
    make_figure(summary,output)
    dependencies={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','matplotlib','threadpoolctl')}
    result={'executed_at':datetime.now(timezone.utc).isoformat(),'data':metadata,'source_commit':BASE_COMMIT,
        'source_prediction_sha256':BASE_SHA,'new_model_fits':0,'benchmark_reuse':'Exploratory; October 11 outcomes already examined',
        'rule':'Ordinary at zero report age and valid prediction, else calendar-only',
        'seeds':list(SEEDS),'daily_delay_probabilities':{'0':.7,'1':.1,'3':.1,'6':.1},
        'unique_target_rows':2208,'overlapping_replays_not_independent_samples':True,
        'command':'python fallback_policy.py --output experiments/2026-10-12',
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'dependencies':dependencies},
        'checks':checks,'mixed_summary':summary.to_dict('records'),
        'limitations':['Same previously exposed city/quarter; not independent validation.',
            'Synthetic daily delays; not measured real-world outage frequency.',
            'Report timestamps assumed correct; no >23h delay evaluation.',
            'Models and original masking draws frozen; no new model fits or uncertainty intervals.']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False));print(f'Checks passed: {checks["passed"]}');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments'/'2026-10-12')
    parser.add_argument('--cache',type=Path,default=ROOT/'data')
    parser.add_argument('--predictions',type=Path,default=ROOT/'results'/'predictions.csv')
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only: print(json.dumps(run_checks(),indent=2))
    else: run(args.output,args.cache,args.predictions)
