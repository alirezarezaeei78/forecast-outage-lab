"""Configurable missing-history training and a preset, exploratory masking sweep."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import numbers
import platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits,threadpool_info
from experiment import ROOT,OUTAGES,SEED,FOLDS,MODEL_CONFIG,features,load_counts,mask_training,seasonal_fallback,run_checks as causal_checks
import matplotlib.pyplot as plt

FRACTIONS=(0.0,0.25,0.5,0.75,1.0)

def validate_fraction(q):
    if isinstance(q,(bool,np.bool_)) or not isinstance(q,numbers.Real) or not np.isfinite(q) or not 0<=q<=1:
        raise ValueError('mask probability must be a finite number in [0,1]')
    return float(q)

def mask_with_probability(counts,index,seed,q=0.5):
    """Replace causal features for drawn reporting delays; leave targets untouched."""
    q=validate_fraction(q)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('seed must be a nonnegative integer')
    if not isinstance(index,pd.DatetimeIndex) or not index.is_unique or not index.is_monotonic_increasing or not index.isin(counts.index).all():
        raise ValueError('Training times must be unique, ordered and present in the hourly grid')
    x=features(counts).loc[index].copy()
    choices=np.random.default_rng(seed).choice(OUTAGES,size=len(index),p=[1-q,q/3,q/3,q/3])
    for k in OUTAGES[1:]:
        selected=index[choices==k]
        if len(selected):x.loc[selected]=features(counts,k).loc[selected]
    return x,{str(k):int((choices==k).sum()) for k in OUTAGES}

def run_checks():
    checks=causal_checks()['checks'].copy()
    idx=pd.date_range('2020-01-01',periods=300,freq='h');y=pd.Series(np.arange(300,dtype=float),index=idx);train=idx[168:250]
    original=y.copy();plain=features(y).loc[train]
    x,c=mask_with_probability(y,train,42,0);pd.testing.assert_frame_equal(x,plain);assert c['0']==len(train)
    checks.append('Zero masking exactly retains ordinary features and rows')
    x,c=mask_with_probability(y,train,42,1);assert c['0']==0 and sum(c.values())==len(train) and x.lag_1h.isna().all()
    checks.append('Full masking replaces recent dependencies without extra rows')
    new,counts=mask_with_probability(y,train,42,.5);old,old_counts=mask_training(y,train,42)
    pd.testing.assert_frame_equal(new,old);assert counts==old_counts
    checks.append('50 percent recipe exactly matches the original masking API')
    for q in FRACTIONS:
        a,ca=mask_with_probability(y,train,43,q);b,cb=mask_with_probability(y,train,43,q)
        pd.testing.assert_frame_equal(a,b);assert ca==cb and a.index.equals(train) and a.columns.equals(plain.columns)
    pd.testing.assert_series_equal(y,original)
    checks.append('All recipes preserve inputs, target values, row order and feature schema; fixed seeds deterministic')
    changed=y.copy();changed.loc[idx[250:]]=999999
    for q in FRACTIONS:pd.testing.assert_frame_equal(mask_with_probability(y,train,42,q)[0],mask_with_probability(changed,train,42,q)[0])
    checks.append('Future target perturbations cannot change training features')
    for q in (-.1,1.1,np.nan,np.inf,-np.inf,True,'0.5'):
        try:mask_with_probability(y,train,42,q)
        except ValueError:pass
        else:raise AssertionError('Invalid probability accepted')
    checks.append('Out-of-range, nonfinite, boolean and nonnumeric probabilities rejected')
    for seed in (-1,True,1.5):
        try:mask_with_probability(y,train,seed,.5)
        except ValueError:pass
        else:raise AssertionError('Invalid seed accepted')
    checks.append('Invalid seeds rejected')
    for bad in (train[::-1],train.append(train[:1]),pd.DatetimeIndex(['1990-01-01'])):
        try:mask_with_probability(y,bad,42,.5)
        except ValueError:pass
        else:raise AssertionError('Invalid training index accepted')
    checks.append('Unordered, duplicate and out-of-grid training indices rejected')
    # Labels are never passed to the masking draw; hidden dependencies use the shared causal feature function.
    return {'passed':len(checks),'checks':checks}

def score(frame,keys):
    rows=[]
    for values,g in frame.groupby(keys):
        if not isinstance(values,tuple):values=(values,)
        rows.append(dict(zip(keys,values),n=len(g),mae=float(mean_absolute_error(g.actual,g.predicted)),rmse=float(np.sqrt(mean_squared_error(g.actual,g.predicted)))))
    return pd.DataFrame(rows)

def result_matrix(metrics,fractions):
    order=[f'mask_{round(q*100):03d}' for q in fractions]+['calendar_hgb','seasonal_fallback']
    matrix=metrics.pivot(index='model',columns='outage_hours',values='mae').loc[order,list(OUTAGES)]
    labels=[f'{q:.0%} mask probability' for q in fractions]+['Calendar only','Seasonal fallback']
    return matrix,labels

def plot_results(metrics,output,fractions):
    matrix,labels=result_matrix(metrics,fractions)
    plt.rcParams.update({'font.size':15,'text.color':'#183346','axes.labelcolor':'#183346'})
    fig,ax=plt.subplots(figsize=(12,10),dpi=120);fig.subplots_adjust(left=.28,right=.92,top=.9,bottom=.24)
    image=ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,vmax=float(matrix.max().max()),aspect='auto')
    ax.set_xticks(range(4),['0h','1h','3h','6h']);ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Most recent reports unavailable at each forecast origin')
    for i in range(len(labels)):
        for j in range(4):
            v=float(matrix.iloc[i,j]);ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=18,color='white' if v>matrix.max().max()*.65 else '#183346')
    bar=fig.colorbar(image,ax=ax,fraction=.045,pad=.03);bar.set_label('MAE (rentals/hour; lower is better)',fontsize=12)
    fig.text(.05,.16,'UCI Bike Sharing; 2,208 observed target hours, July–September 2012; 1-hour forecasts.',fontsize=12)
    fig.text(.05,.115,'Matched 150-iteration HGB models; one masking draw per fold, seeds 42/43/44.',fontsize=12)
    fig.text(.05,.07,'Exploratory reuse of an already examined benchmark; all preset recipes retained.',fontsize=12)
    fig.text(.05,.025,'Data: doi.org/10.24432/C5W894, CC BY 4.0. No confidence intervals or production validation.',fontsize=11)
    fig.savefig(output/'masking-mae.png');fig.savefig(output/'masking-mae.svg');plt.close(fig)

def run(output,cache,fractions):
    fractions=tuple(validate_fraction(q) for q in fractions)
    if fractions!=FRACTIONS:raise ValueError('Recorded sweep requires the preset 0/.25/.5/.75/1 grid; use the imported API for custom studies')
    output.mkdir(parents=True,exist_ok=True);checks=run_checks();y,metadata=load_counts(cache)
    clean=features(y);prediction_frames=[];details=[];fits=0;thread_info=[]
    for fold,(start,end) in enumerate(FOLDS,1):
        tr=y.index[(y.index>=y.index[0]+pd.Timedelta(hours=168))&(y.index<pd.Timestamp(start))&y.notna()]
        te=y.index[(y.index>=pd.Timestamp(start))&(y.index<pd.Timestamp(end))&y.notna()]
        assert tr.max()<te.min()
        xtest={k:features(y,k).loc[te] for k in OUTAGES}
        with threadpool_limits(limits=2):
            thread_info=[{k:i.get(k) for k in ('user_api','internal_api','prefix','version','num_threads')} for i in threadpool_info()]
            assert all(i['num_threads']==2 for i in thread_info)
            for q in fractions:
                x,counts=mask_with_probability(y,tr,SEED+fold-1,q);assert len(x)==len(tr)
                model=HistGradientBoostingRegressor(**MODEL_CONFIG).fit(x,y.loc[tr]);fits+=1
                for k in OUTAGES:
                    pred=model.predict(xtest[k]);assert np.isfinite(pred).all() and (pred>=0).all()
                    prediction_frames.append(pd.DataFrame({'timestamp':te,'fold':fold,'model':f'mask_{round(q*100):03d}',
                        'outage_hours':k,'actual':y.loc[te].to_numpy(),'predicted':pred}))
                details.append({'fold':fold,'mask_probability':q,'mask_seed':SEED+fold-1,'train_rows':len(tr),'test_rows':len(te),
                    'train_start':str(tr.min()),'train_end':str(tr.max()),'test_start':start,'test_end_exclusive':end,
                    'mask_counts':counts,'actual_mask_fraction':1-counts['0']/len(tr),'iterations':150})
                print(f'Fold {fold}, q={q:.2f}: fitted on {len(tr)} rows; test {len(te)}',flush=True)
            calendar=HistGradientBoostingRegressor(**MODEL_CONFIG).fit(clean.loc[tr,['hour','weekday','month','year']],y.loc[tr]);fits+=1
            for k in OUTAGES:
                for name,pred in [('calendar_hgb',calendar.predict(xtest[k][['hour','weekday','month','year']])),('seasonal_fallback',seasonal_fallback(xtest[k],float(y.loc[tr].median())))]:
                    prediction_frames.append(pd.DataFrame({'timestamp':te,'fold':fold,'model':name,'outage_hours':k,'actual':y.loc[te].to_numpy(),'predicted':pred}))
    pred=pd.concat(prediction_frames,ignore_index=True)
    assert not pred.duplicated(['timestamp','model','outage_hours']).any()
    assert pred.groupby(['model','outage_hours']).size().eq(2208).all()
    assert np.isfinite(pred.predicted).all() and (pred.predicted>=0).all()
    pooled=score(pred,['model','outage_hours']);fold_metrics=score(pred,['fold','model','outage_hours'])
    original=pd.read_csv(ROOT/'results'/'predictions.csv',parse_dates=['timestamp']);parity=[]
    for new_name,old_name in [('mask_000','ordinary_hgb'),('mask_050','outage_aware_hgb')]:
        joined=pred[pred.model==new_name].merge(original[original.model==old_name],on=['timestamp','fold','outage_hours'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(joined)==2208*4 and (joined.actual_new==joined.actual_old).all()
        parity.append({'new_model':new_name,'archived_model':old_name,'max_abs_prediction_difference':float(np.max(np.abs(joined.predicted_new-joined.predicted_old)))})
    checks['checks']+=['Strict train-before-test ordering and unchanged training size in all 18 fits',
        'Same 2,208 rows per model/delay with no duplicate predictions',
        'Finite nonnegative forecasts for every recipe and baseline',
        'Archived ordinary/50 percent target identities aligned; numerical differences recorded']
    checks['passed']=len(checks['checks'])
    pred.to_csv(output/'predictions.csv',index=False);pooled.to_csv(output/'metrics.csv',index=False);fold_metrics.to_csv(output/'fold-metrics.csv',index=False)
    pd.DataFrame(details).to_json(output/'mask-counts.json',orient='records',indent=2)
    plot_results(pooled,output,fractions)
    result={'executed_at':datetime.now(timezone.utc).isoformat(),'data':metadata,'model_config':MODEL_CONFIG,
        'mask_probabilities':list(fractions),'mask_seeds_per_fold':[42,43,44],'fits':fits,'horizon_hours':1,'outages':list(OUTAGES),
        'held_out_rows':2208,'benchmark_reuse':'Exploratory; same already examined July–September 2012 quarter',
        'no_test_based_recipe_selection':True,'training_rows_not_augmented':True,'fold_details':details,
        'original_forecast_parity':parity,'checks':checks,'pooled_metrics':pooled.to_dict('records'),
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'threads':2,'observed_native_pools':thread_info,
            'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','matplotlib','threadpoolctl')}},
        'command':'python masking_sweep.py --output experiments/2026-10-13',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['One previously exposed historical city/quarter; not independent validation.',
            'Synthetic rolling reporting delays; mask frequencies not measured operational rates.',
            'One coupled masking draw per recipe/fold; no confidence intervals.',
            'Established HGB, fixed estimator budget; no production or novel algorithm claim.']}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(pooled.pivot(index='model',columns='outage_hours',values='mae').to_string());print(f'Checks passed: {checks["passed"]}; fits: {fits}',flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'experiments'/'2026-10-13');parser.add_argument('--cache',type=Path,default=ROOT/'data');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:print(json.dumps(run_checks(),indent=2))
    else:run(args.output,args.cache,FRACTIONS)
