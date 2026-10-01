"""
Build master_metrics.csv : one tidy long table of every metric from every run.
Schema: variant, model, domain, arm, param_type, param_value, axis, subset, sigma,
        metric, value, delta, ci_lo, ci_hi, p, n, baseline, source
Every figure and the interactive explorer read from this one file.
Run:  python build_master_db.py
"""
import sys, os, pandas as pd, numpy as np
HOME=os.path.expanduser('~'); sys.path.insert(0, HOME)
ROOT=os.path.join(HOME,'mnt','SPIN-main'); DD=os.path.join(ROOT,'thesis_master_data')
import rf_data
d=rf_data.load_all(); H,CAP,AW,CS,CR,SUBH=d['H'],d['CAP'],d['AW'],d['CS'],d['CR'],d['SUBH']
KEYS=['variant','model','domain','arm','param_type','param_value']
rows=[]
def add(base, metric, value, **extra):
    r={k:base.get(k) for k in KEYS}
    r.update(dict(axis=None,subset=None,sigma=None,metric=metric,value=value,
                  delta=None,ci_lo=None,ci_hi=None,p=None,n=None,baseline=None,source=extra.get('source')))
    for k,v in extra.items():
        if k!='source': r[k]=v
    rows.append(r)

# --- HSIC / normalized HSIC / CKA (per sigma) ---
for _,x in H.iterrows():
    b=x.to_dict()
    add(b,'hsic',x.get('hsic'),sigma=x.get('sigma'),delta=x.get('hsic_delta_pct'),baseline=x.get('hsic_baseline'),
        ci_lo=x.get('hsic_delta_ci_lo'),ci_hi=x.get('hsic_delta_ci_hi'),p=(0.0 if x.get('hsic_delta_significant') else None),source='H')
    if pd.notna(x.get('hsic_normalized')):
        add(b,'hsic_norm',x.get('hsic_normalized'),sigma=x.get('sigma'),delta=x.get('hsic_normalized_delta_pct'),baseline=x.get('hsic_normalized_baseline'),source='H')
    if pd.notna(x.get('cka')):
        add(b,'cka',x.get('cka'),sigma=x.get('sigma'),delta=x.get('cka_delta_pct'),baseline=x.get('cka_baseline'),source='H')
    if pd.notna(x.get('coupled_zeroed')):
        add(b,'coupled_zeroed',x.get('coupled_zeroed'),sigma=x.get('sigma'),source='H')

# --- capability: perplexity ---
for _,x in CAP.iterrows():
    add(x.to_dict(),'perplexity',x.get('ppl'),delta=x.get('ppl_delta_pct'),baseline=x.get('ppl_baseline'),
        subset=x.get('ppl_kind'),source='CAP')
    if pd.notna(x.get('sparsity')): add(x.to_dict(),'sparsity',x.get('sparsity'),source='CAP')

# --- MMLU (separate file) ---
mm=pd.read_csv(os.path.join(DD,'metrics_capability_mmlu.csv'))
for _,x in mm.iterrows():
    add(x.to_dict(),'mmlu',x.get('spin'),delta=x.get('delta'),baseline=x.get('base'),n=x.get('n'),subset=x.get('benchmark'),source='mmlu')

# --- awareness (safe rate) ---
for _,x in AW.iterrows():
    add(x.to_dict(),'safe_rate',x.get('spin_rate'),axis=x.get('axis'),subset=x.get('subset'),
        delta=x.get('delta'),baseline=x.get('base_rate'),ci_lo=x.get('ci_lo'),ci_hi=x.get('ci_hi'),
        p=x.get('mcnemar_p'),n=x.get('n'),improve=x.get('improve'),regress=x.get('regress'),net_flips=x.get('net_flips'),source='AW')

# --- per-round coupling (new selected, n_fp) ---
for _,x in CR.iterrows():
    b={'variant':x['variant'],'model':x['model'],'domain':x['domain'],'arm':'iter','param_type':'n','param_value':x['round']}
    add(b,'new_selected',x.get('new'),source='CR')
    add(b,'n_fp_round',x.get('n_fp'),source='CR')
    add(b,'cumulative_zeroed',x.get('cumulative'),source='CR')
    if pd.notna(x.get('seconds')): add(b,'round_seconds',x.get('seconds'),source='CR')

# --- coupling summary (general vs clinical) ---
for _,x in CS.iterrows():
    b={'variant':'spin','model':x['model'],'domain':x['domain'],'arm':'spin','param_type':None,'param_value':None}
    for col in ['nf','n_p','ng','n_fp','n_fp_notg','code','chance_overlap','ratio_vs_chance']:
        if pd.notna(x.get(col)): add(b,'coupling_'+col,x.get(col),source='CS')

# --- per-subset HSIC ---
for _,x in SUBH.iterrows():
    b={'variant':'spin','model':x['model'],'domain':x['domain'],'arm':'spin','param_type':None,'param_value':None}
    add(b,'hsic_subset',x.get('hsic'),subset=x.get('subset'),sigma=x.get('sigma'),delta=x.get('hsic_delta_pct'),baseline=x.get('hsic_baseline'),n=x.get('n'),source='SUBH')

M=pd.DataFrame(rows)

# --- fold in CORRECTED clinical privacy (override buggy as-run privacy deltas) ---
cc=pd.read_csv(os.path.join(ROOT,'healthcare testing','clinical_privacy_corrected.csv'))
armmap={'Hard SPIN':('spin','spin',None,None),
        **{f'softSPIN a{a:.2f}':('soft','soft','alpha',a) for a in [0,0.1,0.3,0.5,0.7,0.9,1.0]},
        **{f'iterSPIN n{n}':('iter','iter','n',n) for n in [1,2,3,5,10]}}
corr=[]
for _,x in cc.iterrows():
    v,arm,pt,pv=armmap[x['arm']]
    base={'variant':v,'model':'Qwen2-7B-Instruct','domain':'health','arm':arm,'param_type':pt,'param_value':pv}
    for metric,val,pcol in [('safe_rate_priv_corrected',x['priv_delta_strict'],'priv_p_strict'),
                            ('safe_rate_priv_corrected_emptySafe',x['priv_delta_emptySafe'],'priv_p_emptySafe')]:
        r={k:base.get(k) for k in KEYS}; r.update(dict(axis='privacy',subset='pooled',sigma=None,metric=metric,
            value=val,delta=val,ci_lo=None,ci_hi=None,p=x[pcol],n=657,baseline=x['base_rate_corr'],source='clinical_corrected'))
        corr.append(r)
M=pd.concat([M,pd.DataFrame(corr)],ignore_index=True)

M['domain']=M['domain'].replace({'health':'clinical','healthcare':'clinical','general (reference)':'general'})
out=os.path.join(DD,'master_metrics.csv'); M.to_csv(out,index=False)
print('rows:',len(M),'| metrics:',sorted(M.metric.unique()))
print('\nby (domain,variant,metric) coverage:')
print(M.groupby(['domain','variant']).size())
print('\nsaved',out)
