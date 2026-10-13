"""Point-in-time report selection and a synthetic failure audit (reserved backup)."""
from __future__ import annotations
import argparse,json,hashlib,platform,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT
import matplotlib.pyplot as plt

def aware(value):
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:raise ValueError('Timezone-aware timestamp required')
    return t.tz_convert('UTC')

def validate_reports(frame):
    if not isinstance(frame,pd.DataFrame) or not {'event_time','available_at','value'}.issubset(frame.columns):raise ValueError('Required report columns absent')
    reports=frame[['event_time','available_at','value']].copy()
    reports['event_time']=pd.to_datetime([aware(v) for v in reports.event_time],utc=True)
    reports['available_at']=pd.to_datetime([aware(v) for v in reports.available_at],utc=True)
    reports['value']=pd.to_numeric(reports.value,errors='raise')
    if not np.isfinite(reports.value).all() or (reports.value<0).any():raise ValueError('Values must be finite nonnegative counts')
    if (reports.available_at<reports.event_time).any() or reports.duplicated(['event_time','available_at']).any():raise ValueError('Invalid clock or ambiguous report revision')
    return reports

def select_report(frame,origin):
    """Choose newest event then newest available revision; return None if unavailable."""
    reports=validate_reports(frame);t=aware(origin)
    eligible=reports[(reports.event_time<=t)&(reports.available_at<=t)]
    if eligible.empty:return None
    row=eligible.sort_values(['event_time','available_at']).iloc[-1]
    return {'event_time':row.event_time.isoformat(),'available_at':row.available_at.isoformat(),'value':float(row.value)}

def run_checks():
    f=pd.DataFrame({'event_time':pd.to_datetime(['2020-01-01T07:00Z','2020-01-01T08:00Z','2020-01-01T09:00Z','2020-01-01T09:00Z']),
        'available_at':pd.to_datetime(['2020-01-01T07:00Z','2020-01-01T11:00Z','2020-01-01T10:00Z','2020-01-01T11:00Z']),'value':[100,200,300,400]})
    checks=[];original=f.copy(deep=True)
    assert select_report(f,'2020-01-01T09:30Z')['value']==100;checks.append('Event-known but unarrived reports excluded')
    assert select_report(f,'2020-01-01T10:00Z')['value']==300;checks.append('Exact arrival-time boundary is included')
    assert select_report(f.iloc[:3],'2020-01-01T11:00Z')['value']==300;checks.append('Late arrival of older event cannot roll back newest event')
    assert select_report(f,'2020-01-01T11:00Z')['value']==400;checks.append('Newest available correction selected within latest event')
    assert select_report(f,'2020-01-01T06:00Z') is None and select_report(f.iloc[:0],'2020-01-01T06:00Z') is None;checks.append('Before history and empty inputs return no invented report')
    changed=f.copy();changed.loc[1:,'value']=999999
    assert select_report(f,'2020-01-01T09:30Z')==select_report(changed,'2020-01-01T09:30Z');checks.append('Unarrived/future values cannot alter available selection')
    select_report(f,'2020-01-01T11:00Z');pd.testing.assert_frame_equal(f,original);checks.append('Input objects are unchanged')
    bad_clock=f.copy();bad_clock.loc[0,'available_at']=pd.Timestamp('2020-01-01T06:00Z')
    bad_value=f.copy();bad_value.loc[0,'value']=np.nan
    for bad in (f.drop(columns='value'),pd.concat([f,f.iloc[:1]],ignore_index=True),bad_clock,bad_value):
        try:select_report(bad,'2020-01-01T11:00Z')
        except (ValueError,TypeError):pass
        else:raise AssertionError('Invalid report accepted')
    checks.append('Missing schema, duplicate revision, invalid clock and NaN rejected')
    for time in ('2020-01-01T11:00',pd.NaT):
        try:select_report(f,time)
        except ValueError:pass
        else:raise AssertionError('Invalid origin accepted')
    checks.append('Naive and missing origin rejected')
    return {'passed':len(checks),'checks':checks}

def run(output):
    output.mkdir(parents=True,exist_ok=True);checks=run_checks();events=pd.date_range('2020-01-01',periods=120,freq='h',tz='UTC')
    rng=np.random.default_rng(77);delay=rng.integers(0,7,len(events));values=rng.integers(1,500,len(events))
    reports=pd.DataFrame({'event_time':events,'available_at':events+pd.to_timedelta(delay,unit='h'),'value':values})
    rows=[]
    for origin in events:
        naive=reports[reports.event_time<=origin].sort_values('event_time').iloc[-1]
        selected=select_report(reports,origin)
        rows.append({'origin':origin.isoformat(),'naive_event':naive.event_time.isoformat(),'naive_available_at':naive.available_at.isoformat(),
            'naive_unavailable_selection':bool(naive.available_at>origin),'safe_event':selected['event_time'] if selected else None,
            'safe_available_at':selected['available_at'] if selected else None,'safe_value':selected['value'] if selected else None,
            'safe_no_report':selected is None,'safe_unavailable_selection':bool(selected and pd.Timestamp(selected['available_at'])>origin)})
    results=pd.DataFrame(rows);assert len(results)==120 and results.safe_unavailable_selection.sum()==0
    counts=[int(results.naive_unavailable_selection.sum()),int(results.safe_unavailable_selection.sum())]
    fig,ax=plt.subplots(figsize=(10,7),dpi=120);fig.subplots_adjust(left=.16,bottom=.28,top=.93,right=.95)
    ax.bar(['Ignore arrival time','Require available report'],counts,color=['#52677F','#087F8C']);ax.set_ylim(0,120)
    ax.set_ylabel('Unavailable selections (of 120 origins)');ax.spines[['top','right']].set_visible(False)
    for i,c in enumerate(counts):ax.text(i,c+3,str(c),ha='center',fontsize=18)
    fig.text(.08,.16,'Synthetic UTC reports; seed 77; hourly events with 0–6h arrival delay.',fontsize=12)
    fig.text(.08,.11,f'No report yet: {int(results.safe_no_report.sum())} origins. Newest event, then newest available revision.',fontsize=12)
    fig.text(.08,.06,'Correctness/failure audit; no model trained, no forecasting-accuracy claim.',fontsize=12)
    fig.savefig(output/'arrival-failures.png');fig.savefig(output/'arrival-failures.svg');plt.close(fig)
    reports.to_csv(output/'synthetic-reports.csv',index=False);results.to_csv(output/'decisions.csv',index=False)
    record={'executed_at':datetime.now(timezone.utc).isoformat(),'synthetic':True,'seed':77,'origins':120,'naive_unavailable':counts[0],'safe_unavailable':counts[1],
        'safe_no_report':int(results.safe_no_report.sum()),'checks':checks,'model_fits':0,'input_sha256':hashlib.sha256((output/'synthetic-reports.csv').read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},
        'command':'python arrival_audit.py --output backups/arrival-audit','status':'Verified unused backup; not posted to LinkedIn',
        'limits':['Synthetic counts/delays; no business data or forecast model evaluation.','Timezone-aware reports and trustworthy arrival clocks required.','In-memory correctness reference; no speed claim or distributed ingestion validation.']}
    (output/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'backups'/'arrival-audit');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:print(json.dumps(run_checks(),indent=2))
    else:run(args.output)
