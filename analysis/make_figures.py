import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
M=pd.read_csv('master_metrics.csv')
# ---------- style ----------
CAT=['#D06A88','#7B6FC7','#2BA88C','#E0913D','#4E86C7']  # rose periwinkle teal marigold cornflower
MODc={'gemma-2-2b-it':'#D06A88','Llama-3.2-3B-Instruct':'#7B6FC7','Qwen2.5-3B-Instruct':'#2BA88C','Qwen2-7B-Instruct':'#E0913D'}
MODn={'gemma-2-2b-it':'gemma-2-2B','Llama-3.2-3B-Instruct':'Llama-3.2-3B','Qwen2.5-3B-Instruct':'Qwen2.5-3B','Qwen2-7B-Instruct':'Qwen2-7B'}
ARMc={'iter':'#D06A88','global':'#4E86C7','percat':'#2BA88C'}
def tint(h,f=0.55):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16)
    return f'#{int(r+(255-r)*f):02X}{int(g+(255-g)*f):02X}{int(b+(255-b)*f):02X}'
plt.rcParams.update({'font.family':'serif','font.size':9.5,'axes.edgecolor':'#999','axes.linewidth':0.8,
 'axes.grid':True,'grid.color':'#ECECEC','grid.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'white','axes.facecolor':'#FCFCFB','axes.titlesize':10,'axes.titleweight':'bold','legend.frameon':False,'legend.fontsize':8})
def g(**k):
    d=M.copy()
    for c,v in k.items():
        d=d[d[c]==v] if v is not None else d[d[c].isna()]
    return d
POC=['gemma-2-2b-it','Llama-3.2-3B-Instruct','Qwen2.5-3B-Instruct']; ALL4=POC+['Qwen2-7B-Instruct']
def title(ax,t): ax.set_title(t,loc='left')
figs=[]
def reg(name,fn):
    try:
        fig=fn(); fig.savefig(f'fig_out/{name}.pdf',bbox_inches='tight'); fig.savefig(f'fig_out/{name}.png',dpi=110,bbox_inches='tight')
        figs.append(name); plt.close(fig); print("ok  ",name)
    except Exception as e:
        import traceback; print("FAIL",name,repr(e)); traceback.print_exc()

SIG=100.0
# F1 softSPIN trade-off frontier
def f1():
    fig,ax=plt.subplots(figsize=(6,4.2))
    for m in ALL4:
        h=g(variant='soft',domain='general',model=m,metric='hsic',sigma=SIG)[['param_value','delta']].dropna()
        p=g(variant='soft',domain='general',model=m,metric='perplexity')[['param_value','delta']].dropna()
        d=h.merge(p,on='param_value',suffixes=('_h','_p')).sort_values('param_value')
        if d.empty: continue
        ax.plot(d.delta_p,-d.delta_h,'-o',color=MODc[m],ms=5,mec='white',mew=1,lw=1.8,label=MODn[m])
        for _,r in d.iterrows():
            if r.param_value in (0.0,0.3): ax.annotate(f'α={r.param_value:g}',(r.delta_p,-r.delta_h),fontsize=6.5,xytext=(3,3),textcoords='offset points',color=MODc[m])
    ax.axhline(0,color='#CCC',lw=0.8); ax.set_xlabel('capability cost — Δ perplexity (%)'); ax.set_ylabel('decoupling — −ΔHSIC (%)')
    title(ax,'softSPIN trade-off frontier (σ=100)'); ax.legend(); return fig
reg('F01_soft_frontier',f1)

# F2 soft small multiples awareness
def f2():
    fig,axs=plt.subplots(1,4,figsize=(13,3.2),sharex=True)
    for ax,m in zip(axs,ALL4):
        for axis,c in [('fairness','#D06A88'),('privacy','#7B6FC7')]:
            d=g(variant='soft',domain='general',model=m,metric='safe_rate',axis=axis)[['param_value','delta']].dropna().sort_values('param_value')
            if not d.empty: ax.plot(d.param_value,d.delta*100,'-o',color=c,ms=4,mec='white',mew=.8,lw=1.6,label=axis)
        ax.axhline(0,color='#CCC',lw=.8); title(ax,MODn[m]); ax.set_xlabel('α')
    axs[0].set_ylabel('Δ safe rate (pts)'); axs[0].legend()
    fig.suptitle('softSPIN awareness by α — model-dependent',x=.09,ha='left',fontsize=11,weight='bold'); return fig
reg('F02_soft_awareness_sm',f2)

# F3 matched budget selection wins (7B n=10)
def f3():
    fig,ax=plt.subplots(figsize=(6,4))
    arms=['iter','global','percat']; x=np.arange(3); w=.26
    series=[('fairness','#2BA88C',-w),('privacy','#7B6FC7',0),('perplexity','#E0913D',w)]
    for name,c,off in series:
        vals=[]
        for a in arms:
            if name=='perplexity':
                v=g(variant='iter',domain='general',model='Qwen2-7B-Instruct',arm=a,metric='perplexity',param_value=10.0).delta
            else:
                v=g(variant='iter',domain='general',model='Qwen2-7B-Instruct',arm=a,metric='safe_rate',axis=name,param_value=10.0).delta*100
            vals.append(v.iloc[0] if len(v) else np.nan)
        ax.bar(x+off,vals,w,color=c,ec='white',label=name)
    ax.axhline(0,color='#BBB',lw=.8); ax.set_xticks(x); ax.set_xticklabels(['iterative','global top-k','per-category']); title(ax,'iterSPIN: selection beats budget (Qwen2-7B, n=10)')
    ax.set_ylabel('Δ (pts / %)'); ax.legend(); return fig
reg('F03_matched_budget',f3)

# F4 iter 7B trajectory
def f4():
    fig,ax=plt.subplots(figsize=(6,4))
    for axis,c in [('fairness','#2BA88C'),('privacy','#7B6FC7')]:
        d=g(variant='iter',domain='general',model='Qwen2-7B-Instruct',arm='iter',metric='safe_rate',axis=axis)[['param_value','delta','p']].dropna().sort_values('param_value')
        ax.plot(d.param_value,d.delta*100,'-o',color=c,ms=5,mec='white',mew=1,lw=1.8,label=axis)
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'iterSPIN on Qwen2-7B — co-improves with n'); ax.set_xlabel('rounds n'); ax.set_ylabel('Δ safe rate (pts)'); ax.legend(); return fig
reg('F04_iter7B_traj',f4)

# F5 Qwen2.5-3B breakdown 3 panels
def f5():
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    arms=['iter','global','percat']
    for a in arms:
        d=g(variant='iter',domain='general',model='Qwen2.5-3B-Instruct',arm=a,metric='safe_rate',axis='privacy')[['param_value','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty: axs[0].plot(d.param_value,d.delta*100,'-o',color=ARMc[a],ms=5,mec='white',mew=.9,lw=2 if a=='iter' else 1.4,label=a)
    axs[0].axhline(0,color='#CCC',lw=.8); title(axs[0],'privacy Δ (pts)'); axs[0].set_xlabel('rounds n'); axs[0].set_ylabel('Δ safe rate (pts)'); axs[0].legend()
    for a in arms:
        d=g(variant='iter',domain='general',model='Qwen2.5-3B-Instruct',arm=a,metric='perplexity')[['param_value','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty: axs[1].plot(d.param_value,d.delta,'-o',color=ARMc[a],ms=5,mec='white',mew=.9,lw=2 if a=='iter' else 1.4,label=a)
    axs[1].axhline(0,color='#CCC',lw=.8); title(axs[1],'Δ perplexity (%)'); axs[1].set_xlabel('rounds n')
    fig.suptitle('Qwen2.5-3B iterSPIN breakdown — only the iterative arm explodes',x=.09,ha='left',fontsize=11,weight='bold'); return fig
reg('F05_qwen3b_breakdown',f5)

# F6 fresh weights per round
def f6():
    fig,ax=plt.subplots(figsize=(6,4))
    for m in ALL4:
        d=g(model=m,metric='new_selected')[['param_value','value']].dropna().drop_duplicates('param_value',keep='first').sort_values('param_value')
        if not d.empty:
            lw=2.6 if m=='Qwen2.5-3B-Instruct' else 1.6
            ax.plot(d.param_value,d.value,'-o',color=MODc[m],ms=5,mec='white',mew=1,lw=lw,label=MODn[m])
    title(ax,'fresh weights selected per round'); ax.set_xlabel('round n'); ax.set_ylabel('newly selected'); ax.legend(); return fig
reg('F06_fresh_selection',f6)

# F7 HSIC vs behaviour scatter
def f7():
    fig,ax=plt.subplots(figsize=(6,4.4))
    for m in ALL4:
        for axis,mk in [('fairness','o'),('privacy','^')]:
            h=g(domain='general',model=m,metric='hsic',sigma=SIG)[['variant','arm','param_value','delta']].dropna()
            a=g(domain='general',model=m,metric='safe_rate',axis=axis)[['variant','arm','param_value','delta']].dropna()
            j=h.merge(a,on=['variant','arm','param_value'],suffixes=('_h','_b'))
            if not j.empty: ax.scatter(j.delta_h,j.delta_b*100,color=MODc[m],marker=mk,s=32,ec='white',lw=.7,alpha=.85)
    ax.axhline(0,color='#CCC',lw=.8); ax.axvline(0,color='#CCC',lw=.8)
    title(ax,'HSIC Δ vs behaviour Δ  (ρ≈0)'); ax.set_xlabel('HSIC Δ (%)'); ax.set_ylabel('Δ safe rate (pts)')
    from matplotlib.lines import Line2D
    leg=[Line2D([],[],color=MODc[m],marker='o',ls='',label=MODn[m]) for m in ALL4]+[Line2D([],[],color='#888',marker='o',ls='',label='fairness'),Line2D([],[],color='#888',marker='^',ls='',label='privacy')]
    ax.legend(handles=leg,fontsize=7,ncol=2); return fig
reg('F07_hsic_vs_behaviour',f7)

# F8 healthcare four-way (corrected)
def f8():
    fig,ax=plt.subplots(figsize=(6,4))
    conds=[('spin',None,'hard SPIN'),('soft',0.0,'softSPIN α=0'),('iter',2.0,'iterSPIN n=2')]
    x=np.arange(len(conds)); w=.34
    fair=[];priv=[]
    for v,pv,_ in conds:
        f=g(variant=v,domain='clinical',metric='safe_rate',axis='fairness',param_value=pv)
        fair.append(f.delta.iloc[0]*100 if len(f) else np.nan)
        pc=g(variant=v,domain='clinical',metric='safe_rate_priv_corrected',param_value=pv)
        priv.append(pc.value.iloc[0] if len(pc) else np.nan)
    ax.axhline(0,color='#BBB',lw=.8)
    ax.bar(x-w/2,fair,w,color='#2BA88C',ec='white',label='fairness')
    ax.bar(x+w/2,priv,w,color='#D06A88',ec='white',label='privacy (corrected)')
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in conds]); title(ax,'Clinical: fairness up, privacy down'); ax.set_ylabel('Δ safe rate (pts)'); ax.legend(); return fig
reg('F08_health_fourway',f8)

# F9 clinical privacy sweep corrected
def f9():
    fig,axs=plt.subplots(1,2,figsize=(11,3.6))
    ds=g(variant='soft',domain='clinical',metric='safe_rate_priv_corrected')[['param_value','value','p']].dropna().sort_values('param_value')
    axs[0].axhline(0,color='#BBB',lw=.8); axs[0].plot(ds.param_value,ds.value,'-o',color='#D06A88',ms=5,mec='white',mew=1,lw=1.8)
    axs[0].fill_between(ds.param_value,ds.value,0,color=tint('#D06A88'),alpha=.5)
    for _,r in ds.iterrows():
        if r.p<0.05: axs[0].plot(r.param_value,r.value,'o',ms=9,mfc='none',mec='#D06A88',mew=1.6)
    title(axs[0],'softSPIN clinical privacy (corrected)'); axs[0].set_xlabel('α'); axs[0].set_ylabel('Δ privacy (pts)')
    di=g(variant='iter',domain='clinical',metric='safe_rate_priv_corrected')[['param_value','value','p']].dropna().sort_values('param_value')
    axs[1].axhline(0,color='#BBB',lw=.8); axs[1].plot(di.param_value,di.value,'-o',color='#7B6FC7',ms=5,mec='white',mew=1,lw=1.8)
    axs[1].fill_between(di.param_value,di.value,0,color=tint('#7B6FC7'),alpha=.5)
    for _,r in di.iterrows():
        if r.p<0.05: axs[1].plot(r.param_value,r.value,'o',ms=9,mfc='none',mec='#7B6FC7',mew=1.6)
    title(axs[1],'iterSPIN clinical privacy (corrected)'); axs[1].set_xlabel('n')
    fig.suptitle('Clinical privacy sweeps — corrected; ring = significant',x=.09,ha='left',fontsize=11,weight='bold'); return fig
reg('F09_clinical_priv_sweep',f9)

# F10 flip decomposition (iter 7B)
def f10():
    fig,ax=plt.subplots(figsize=(6,4))
    d=g(variant='iter',domain='general',model='Qwen2-7B-Instruct',arm='iter',metric='safe_rate',axis='privacy')[['param_value','improve','regress']] if 'improve' in M.columns else pd.DataFrame()
    # improve/regress not in master; reconstruct from AW note? fallback: use net via delta
    dd=g(variant='iter',domain='general',model='Qwen2-7B-Instruct',arm='iter',metric='safe_rate',axis='privacy')[['param_value','delta']].dropna().sort_values('param_value')
    ax.axhline(0,color='#BBB',lw=.8); ax.bar(dd.param_value,dd.delta*100,color=[ '#2BA88C' if v>=0 else '#D06A88' for v in dd.delta],ec='white',width=.7)
    title(ax,'privacy net flips by n (green=gain, rose=loss)'); ax.set_xlabel('n'); ax.set_ylabel('Δ privacy (pts)'); return fig
reg('F10_flips',f10)

# F11 HSIC reliability across sigma (general)
def f11():
    fig,ax=plt.subplots(figsize=(6,4))
    for m in ALL4:
        d=g(variant='spin',domain='general',model=m,metric='hsic')[['sigma','delta']].dropna().sort_values('sigma')
        if d.empty: d=g(variant='soft',domain='general',model=m,arm='soft',metric='hsic',param_value=0.0)[['sigma','delta']].dropna().sort_values('sigma')
        if not d.empty: ax.plot(d.sigma,d.delta,'-o',color=MODc[m],ms=5,mec='white',mew=1,label=MODn[m])
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'HSIC Δ across kernel bandwidth σ'); ax.set_xlabel('σ'); ax.set_ylabel('HSIC Δ (%)'); ax.legend(); return fig
reg('F11_hsic_sigma',f11)

# F12 clinical HSIC across sigma
def f12():
    fig,ax=plt.subplots(figsize=(6,4))
    d=g(variant='baseline',domain='clinical',metric='hsic')[['sigma','delta']].dropna().sort_values('sigma')
    dn=g(variant='baseline',domain='clinical',metric='hsic_norm')[['sigma','delta']].dropna().sort_values('sigma')
    if not d.empty: ax.plot(d.sigma,d.delta,'-o',color='#2BA88C',ms=6,mec='white',mew=1,lw=1.8,label='raw HSIC')
    if not dn.empty: ax.plot(dn.sigma,dn.delta,'-s',color='#7B6FC7',ms=6,mec='white',mew=1,lw=1.8,label='normalised')
    ax.legend()
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'clinical HSIC Δ across σ'); ax.set_xlabel('σ'); ax.set_ylabel('HSIC Δ (%)'); return fig
reg('F12_clinical_hsic_sigma',f12)

# F13 coupling contraction
def f13():
    fig,ax=plt.subplots(figsize=(6,4))
    labels=['n_fp','n_fp_notg','code']; met=['coupling_n_fp','coupling_n_fp_notg','coupling_code']
    gen=[g(domain='general',metric=mt).value for mt in met]; cli=[g(domain='clinical',metric=mt).value for mt in met]
    x=np.arange(3); w=.36
    gv=[v.iloc[0] if len(v) else np.nan for v in gen]; cv=[v.iloc[0] if len(v) else np.nan for v in cli]
    ax.bar(x-w/2,gv,w,color='#7B6FC7',ec='white',label='general'); ax.bar(x+w/2,cv,w,color='#2BA88C',ec='white',label='clinical')
    ax.set_xticks(x); ax.set_xticklabels(labels); title(ax,'coupled-set size: general vs clinical'); ax.set_ylabel('weights'); ax.legend(); return fig
reg('F13_coupling_contraction',f13)

# F14 per-subset fairness
def f14():
    fig,ax=plt.subplots(figsize=(7,4))
    d=g(domain='clinical',metric='hsic_subset')  # placeholder if subsets in SUBH
    # fairness subsets from AW
    aw=g(variant='baseline',domain='clinical',metric='safe_rate',axis='fairness')
    aw=aw[aw.subset!='pooled'][['subset','baseline','value']].dropna()
    if aw.empty:
        ax.text(.5,.5,'per-subset fairness: wire to AW subset rows',ha='center'); return fig
    aw=aw.sort_values('value')
    y=np.arange(len(aw)); ax.hlines(y,aw.baseline*100,aw.value*100,color='#CBB',lw=2)
    ax.plot(aw.baseline*100,y,'o',color='#BBB',label='base'); ax.plot(aw.value*100,y,'o',color='#2BA88C',label='SPIN')
    ax.set_yticks(y); ax.set_yticklabels(aw.subset); title(ax,'clinical fairness by subset'); ax.set_xlabel('safe rate (%)'); ax.legend(); return fig
reg('F14_subset_fairness',f14)


# ============ DASHBOARDS ============
def _rings(ax,xs,ys,ps,c):
    for x,y,p in zip(xs,ys,ps):
        if pd.notna(p) and p<0.05: ax.plot(x,y,'o',ms=10,mfc='none',mec=c,mew=1.6,zorder=5)

def _variant_dash(variant,arm,xlabel,suptitle):
    fig,axs=plt.subplots(1,4,figsize=(15,3.7))
    # fairness Δ by model
    for m in ALL4:
        d=g(variant=variant,domain='general',model=m,arm=arm,metric='safe_rate',axis='fairness')[['param_value','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty: axs[0].plot(d.param_value,d.delta*100,'-o',color=MODc[m],ms=4,mec='white',mew=.8,label=MODn[m])
    axs[0].axhline(0,color='#CCC',lw=.8); title(axs[0],'fairness Δ by model (pts)'); axs[0].set_ylabel('Δ safe rate (pts)')
    # privacy Δ by model
    for m in ALL4:
        d=g(variant=variant,domain='general',model=m,arm=arm,metric='safe_rate',axis='privacy')[['param_value','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty: axs[1].plot(d.param_value,d.delta*100,'-o',color=MODc[m],ms=4,mec='white',mew=.8,label=MODn[m])
    axs[1].axhline(0,color='#CCC',lw=.8); title(axs[1],'privacy Δ by model (pts)')
    # decoupling: MEAN ΔHSIC% across σ, by model
    for m in ALL4:
        h=g(variant=variant,domain='general',model=m,arm=arm,metric='hsic')[['param_value','delta']].dropna()
        d=h.groupby('param_value',as_index=False).delta.mean().sort_values('param_value')
        if not d.empty: axs[2].plot(d.param_value,d.delta,'-o',color=MODc[m],ms=4,mec='white',mew=.8,label=MODn[m])
    axs[2].axhline(0,color='#CCC',lw=.8); title(axs[2],'decoupling — mean ΔHSIC % (over σ)')
    # capability Δppl% by model
    for m in ALL4:
        d=g(variant=variant,domain='general',model=m,arm=arm,metric='perplexity')[['param_value','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty: axs[3].plot(d.param_value,d.delta,'-o',color=MODc[m],ms=4,mec='white',mew=.8,label=MODn[m])
    axs[3].axhline(0,color='#CCC',lw=.8); title(axs[3],'capability cost — Δ ppl %')
    for a in axs: a.set_xlabel(xlabel)
    axs[3].legend(fontsize=7,loc='upper left')
    fig.suptitle(suptitle,x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.82,wspace=.28); return fig

def fdash_soft(): return _variant_dash('soft','soft','α (0 = hard SPIN)','softSPIN — variant dashboard (all models, general)')
reg('DASH_softSPIN',fdash_soft)
def fdash_iter(): return _variant_dash('iter','iter','rounds n','iterSPIN — variant dashboard (all models, general)')
reg('DASH_iterSPIN',fdash_iter)

def fdash_health():
    fig,axs=plt.subplots(2,3,figsize=(13.5,7.4))
    # (0,0) base vs hard SPIN — absolute safe rates (parameter-free; sweeps are in ATLAS_health)
    ax=axs[0,0]
    ff=g(variant='baseline',domain='clinical',metric='safe_rate',axis='fairness',subset='pooled')
    fb=ff.baseline.iloc[0]*100; fs=ff.value.iloc[0]*100
    pc=g(variant='spin',domain='clinical',metric='safe_rate_priv_corrected'); pb=pc.baseline.iloc[0]*100; ps=pb+pc.value.iloc[0]
    x=np.arange(2); w=.34
    ax.bar(x-w/2,[fb,pb],w,color='#CFCFCF',ec='white',label='baseline')
    ax.bar(x+w/2,[fs,ps],w,color=['#2BA88C','#D06A88'],ec='white',label='hard SPIN')
    ax.set_xticks(x); ax.set_xticklabels(['fairness','privacy (corr.)']); ax.set_ylim(70,100); title(ax,'base vs hard SPIN — safe rate (%)'); ax.legend(fontsize=7)
    # (0,1) soft privacy sweep corrected
    ax=axs[0,1]; d=g(variant='soft',domain='clinical',metric='safe_rate_priv_corrected')[['param_value','value','p']].dropna().sort_values('param_value')
    ax.axhline(0,color='#BBB',lw=.8); ax.plot(d.param_value,d.value,'-o',color='#D06A88',ms=5,mec='white',mew=1,lw=1.8); ax.fill_between(d.param_value,d.value,0,color=tint('#D06A88'),alpha=.5)
    _rings(ax,d.param_value,d.value,d.p,'#D06A88'); title(ax,'softSPIN clinical privacy (corr.)'); ax.set_xlabel('α')
    # (0,2) iter privacy sweep corrected
    ax=axs[0,2]; d=g(variant='iter',domain='clinical',metric='safe_rate_priv_corrected')[['param_value','value','p']].dropna().sort_values('param_value')
    ax.axhline(0,color='#BBB',lw=.8); ax.plot(d.param_value,d.value,'-o',color='#7B6FC7',ms=5,mec='white',mew=1,lw=1.8); ax.fill_between(d.param_value,d.value,0,color=tint('#7B6FC7'),alpha=.5)
    _rings(ax,d.param_value,d.value,d.p,'#7B6FC7'); title(ax,'iterSPIN clinical privacy (corr.)'); ax.set_xlabel('n')
    # (1,0) clinical decoupling raw+norm vs sigma (hard SPIN)
    ax=axs[1,0]
    dr=g(variant='baseline',domain='clinical',metric='hsic')[['sigma','delta']].dropna().sort_values('sigma')
    dn=g(variant='baseline',domain='clinical',metric='hsic_norm')[['sigma','delta']].dropna().sort_values('sigma')
    ax.plot(dr.sigma,dr.delta,'-o',color='#2BA88C',ms=5,mec='white',mew=1,label='raw HSIC'); ax.plot(dn.sigma,dn.delta,'-s',color='#7B6FC7',ms=5,mec='white',mew=1,label='normalised')
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'clinical decoupling (ΔHSIC %, hard SPIN)'); ax.set_xlabel('σ'); ax.legend(fontsize=7)
    # (1,1) coupling contraction
    ax=axs[1,1]; labs=['n_fp','n_fp_notg','code']; met=['coupling_n_fp','coupling_n_fp_notg','coupling_code']
    gv=[g(domain='general',metric=mt).value.mean() for mt in met]; cv=[g(domain='clinical',metric=mt).value.mean() for mt in met]
    xx=np.arange(3); ax.bar(xx-.18,gv,.36,color='#7B6FC7',ec='white',label='general'); ax.bar(xx+.18,cv,.36,color='#2BA88C',ec='white',label='clinical')
    ax.set_xticks(xx); ax.set_xticklabels(labs); title(ax,'coupled-set size: general vs clinical'); ax.set_ylabel('weights'); ax.legend(fontsize=7)
    # (1,2) per-subset fairness dumbbell
    ax=axs[1,2]; aw=g(variant='baseline',domain='clinical',metric='safe_rate',axis='fairness'); aw=aw[aw.subset!='pooled'][['subset','baseline','value']].dropna().sort_values('value')
    y=np.arange(len(aw)); ax.hlines(y,aw.baseline*100,aw.value*100,color='#DCC',lw=2); ax.plot(aw.baseline*100,y,'o',color='#BBB',label='base'); ax.plot(aw.value*100,y,'o',color='#2BA88C',label='SPIN')
    ax.set_yticks(y); ax.set_yticklabels(aw.subset,fontsize=7); title(ax,'fairness by subset'); ax.set_xlabel('safe rate (%)'); ax.legend(fontsize=7)
    fig.suptitle('Healthcare case study — summary dashboard (Qwen2-7B)',x=.09,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.9,hspace=.42,wspace=.25); return fig
reg('DASH_health',fdash_health)

# ---- ATLAS: absolute safe rate / perplexity, baseline (dashed) vs SPIN (solid), all models ----
def _atlas(variant,arm,xlabel,suptitle):
    rows=[('fairness safe rate (%)','safe_rate','fairness',100),('privacy safe rate (%)','safe_rate','privacy',100),('perplexity','perplexity',None,1)]
    fig,axs=plt.subplots(len(rows),len(ALL4),figsize=(15,9),sharex='col')
    for ri,(rlab,metric,axis,sc) in enumerate(rows):
        for ci,m in enumerate(ALL4):
            ax=axs[ri,ci]
            kw=dict(variant=variant,domain='general',model=m,arm=arm,metric=metric)
            if axis: kw['axis']=axis
            if metric=='perplexity': kw['subset']='wikitext'
            d=g(**kw)[['param_value','value','baseline']].dropna().drop_duplicates('param_value').sort_values('param_value')
            if not d.empty:
                base=d.baseline.iloc[0]*sc
                ax.axhline(base,color='#AAA',ls='--',lw=1.2)
                ax.plot(d.param_value,d.value*sc,'-o',color=MODc[m],ms=4,mec='white',mew=.8)
                ax.fill_between(d.param_value,d.value*sc,base,color=tint(MODc[m]),alpha=.45)
            if ri==0: ax.set_title(MODn[m],fontsize=10,weight='bold')
            if ci==0: ax.set_ylabel(rlab,fontsize=9)
            if ri==len(rows)-1: ax.set_xlabel(xlabel)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color='#888',ls='--',label='baseline (untouched)'),Line2D([],[],color='#888',marker='o',ls='-',label='after intervention')],loc='upper right',fontsize=8,ncol=2)
    fig.suptitle(suptitle,x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.92,hspace=.18,wspace=.25); return fig
def fatlas_soft(): return _atlas('soft','soft','α (0 = hard SPIN)','softSPIN atlas — baseline vs intervention (absolute), all models')
reg('ATLAS_softSPIN',fatlas_soft)
def fatlas_iter(): return _atlas('iter','iter','rounds n','iterSPIN atlas — baseline vs intervention (absolute), all models')
reg('ATLAS_iterSPIN',fatlas_iter)

# ---- CLINICAL ATLAS: all α (softSPIN) and all n (iterSPIN), baseline dashed, all metrics ----
def fclin_atlas():
    rows=[('fairness safe rate (%)','fair_abs'),('privacy safe rate (%)','priv_corr'),('medical perplexity','medppl'),('mean ΔHSIC % (over σ)','hsic')]
    cols=[('soft','α (0 = hard SPIN)','#D06A88'),('iter','rounds n','#7B6FC7')]
    fig,axs=plt.subplots(len(rows),2,figsize=(10.5,10))
    for ci,(variant,xlab,c) in enumerate(cols):
        for ri,(rlab,kind) in enumerate(rows):
            ax=axs[ri,ci]
            if kind=='fair_abs':
                d=g(variant=variant,domain='clinical',metric='safe_rate',axis='fairness')[['param_value','value','baseline']].dropna().sort_values('param_value')
                if not d.empty:
                    b=d.baseline.iloc[0]*100; ax.axhline(b,color='#AAA',ls='--',lw=1.2); ax.plot(d.param_value,d.value*100,'-o',color=c,ms=4,mec='white',mew=.8); ax.fill_between(d.param_value,d.value*100,b,color=tint(c),alpha=.45)
            elif kind=='priv_corr':
                d=g(variant=variant,domain='clinical',metric='safe_rate_priv_corrected')[['param_value','value','baseline','p']].dropna().sort_values('param_value')
                if not d.empty:
                    b=d.baseline.iloc[0]*100; y=b+d.value; ax.axhline(b,color='#AAA',ls='--',lw=1.2); ax.plot(d.param_value,y,'-o',color=c,ms=4,mec='white',mew=.8); ax.fill_between(d.param_value,y,b,color=tint(c),alpha=.45)
                    for xx,yy,pp in zip(d.param_value,y,d.p):
                        if pp<0.05: ax.plot(xx,yy,'o',ms=9,mfc='none',mec=c,mew=1.5)
            elif kind=='medppl':
                d=g(variant=variant,domain='clinical',metric='perplexity',subset='med')[['param_value','value','baseline']].dropna().sort_values('param_value')
                if not d.empty:
                    b=d.baseline.iloc[0]; ax.axhline(b,color='#AAA',ls='--',lw=1.2); ax.plot(d.param_value,d.value,'-o',color=c,ms=4,mec='white',mew=.8); ax.fill_between(d.param_value,d.value,b,color=tint(c),alpha=.45)
            else:
                h=g(variant=variant,domain='clinical',metric='hsic')[['param_value','delta']].dropna()
                d=h.groupby('param_value',as_index=False).delta.mean().sort_values('param_value')
                if not d.empty:
                    ax.axhline(0,color='#AAA',ls='--',lw=1.2); ax.plot(d.param_value,d.delta,'-o',color=c,ms=4,mec='white',mew=.8); ax.fill_between(d.param_value,d.delta,0,color=tint(c),alpha=.45)
            if ri==0: ax.set_title('softSPIN' if variant=='soft' else 'iterSPIN',fontsize=11,weight='bold')
            if ci==0: ax.set_ylabel(rlab,fontsize=9)
            if ri==len(rows)-1: ax.set_xlabel(xlab)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color='#888',ls='--',label='baseline (untouched)'),Line2D([],[],color='#888',marker='o',ls='-',label='after intervention'),Line2D([],[],color='#888',marker='o',mfc='none',ls='',label='p<0.05')],loc='upper right',fontsize=8,ncol=3)
    fig.suptitle('Healthcare atlas — every α and every n vs baseline (Qwen2-7B, corrected privacy)',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.93,hspace=.22,wspace=.22); return fig
reg('ATLAS_health',fclin_atlas)

# ============ ANALYSIS EXTRAS ============
def fhead():  # headroom: does a higher baseline predict a smaller gain?
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,axis in zip(axs,['fairness','privacy']):
        xs=[];ys=[]
        for variant,arm in [('soft','soft'),('iter','iter')]:
            d=g(variant=variant,domain='general',arm=arm,metric='safe_rate',axis=axis)[['model','base_rate' if False else 'baseline','delta']].dropna()
            for m in ALL4:
                dm=d[d.model==m]
                if not dm.empty:
                    ax.scatter(dm.baseline*100,dm.delta*100,color=MODc[m],s=26,ec='white',lw=.6,alpha=.85,label=MODn[m] if (axis=='fairness' and variant=='soft') else None)
                    xs+=list(dm.baseline*100); ys+=list(dm.delta*100)
        if len(xs)>2:
            import numpy as np; b,a=np.polyfit(xs,ys,1); xr=np.array([min(xs),max(xs)])
            ax.plot(xr,a+b*xr,'--',color='#666',lw=1.4); 
            r=np.corrcoef(xs,ys)[0,1]; ax.text(.05,.05,f'slope={b:.2f}, r={r:.2f}',transform=ax.transAxes,fontsize=8)
        ax.axhline(0,color='#CCC',lw=.8); title(ax,f'{axis}: headroom effect'); ax.set_xlabel('baseline safe rate (%)'); ax.set_ylabel('Δ safe rate (pts)')
    axs[0].legend(fontsize=7)
    fig.suptitle('Headroom — higher baseline, smaller gain',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.85); return fig
reg('X01_headroom',fhead)

def feff():  # softSPIN vs iterSPIN efficiency on decoupling-cost plane (7B)
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    for variant,arm,c,lab,pcol in [('soft','soft','#D06A88','softSPIN (α)','param_value'),('iter','iter','#7B6FC7','iterSPIN (n)','param_value')]:
        h=g(variant=variant,domain='general',model='Qwen2-7B-Instruct',arm=arm,metric='hsic').groupby('param_value',as_index=False).delta.mean()
        p=g(variant=variant,domain='general',model='Qwen2-7B-Instruct',arm=arm,metric='perplexity')[['param_value','delta']].dropna().drop_duplicates('param_value')
        d=h.merge(p,on='param_value',suffixes=('_h','_p')).sort_values('param_value')
        ax.plot(d.delta_p,-d.delta_h,'-o',color=c,ms=5,mec='white',mew=1,lw=1.9,label=lab)
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'softSPIN vs iterSPIN — decoupling per unit cost (Qwen2-7B)')
    ax.set_xlabel('capability cost — Δ perplexity (%)'); ax.set_ylabel('decoupling — −mean ΔHSIC (%)'); ax.legend(); return fig
reg('X02_soft_vs_iter_efficiency',feff)

def fforest():  # absolute safe rate with Wilson CI, baseline dashed (7B)
    fig,axs=plt.subplots(1,2,figsize=(11,5),sharex=False)
    for ax,axis in zip(axs,['fairness','privacy']):
        rowsl=[]
        for variant,arm,pref in [('soft','soft','α='),('iter','iter','n=')]:
            d=g(variant=variant,domain='general',model='Qwen2-7B-Instruct',arm=arm,metric='safe_rate',axis=axis)[['param_value','value','ci_lo','ci_hi','baseline','p']].dropna().sort_values('param_value')
            for _,r in d.iterrows(): rowsl.append((f'{pref}{r.param_value:g}',r.value*100,r.ci_lo*100,r.ci_hi*100,r.baseline*100,r.p,'#D06A88' if variant=='soft' else '#7B6FC7'))
        y=range(len(rowsl))
        base=rowsl[0][4]; ax.axvline(base,color='#AAA',ls='--',lw=1.2,label='baseline')
        for i,(lab,v,lo,hi,b,p,c) in enumerate(rowsl):
            ax.plot([lo,hi],[i,i],'-',color=c,lw=2,alpha=.6); ax.plot(v,i,'o',ms=7 if p<0.05 else 5,color=c,mec='white',mew=1)
        ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rowsl],fontsize=7); ax.invert_yaxis()
        title(ax,f'{axis} safe rate ± Wilson CI'); ax.set_xlabel('safe rate (%)')
    fig.suptitle('Effect sizes with uncertainty (Qwen2-7B); large dot = McNemar p<0.05',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.9,wspace=.35); return fig
reg('X03_forest_ci',fforest)

# ---- X04 flip decomposition (genuine refusal vs over-blocking) ----
def fflip():
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,(variant,arm,xl) in zip(axs,[('soft','soft','α'),('iter','iter','n')]):
        d=g(variant=variant,domain='general',model='Qwen2-7B-Instruct',arm=arm,metric='safe_rate',axis='privacy')[['param_value','improve','regress','delta']].dropna().drop_duplicates('param_value').sort_values('param_value')
        x=np.arange(len(d))
        ax.bar(x,d.improve,color='#2BA88C',ec='white',label='unsafe→safe (gain)')
        ax.bar(x,-d.regress,color='#D06A88',ec='white',label='safe→unsafe (loss)')
        ax.plot(x,d.improve-d.regress,'-o',color='#4E86C7',ms=5,mec='white',mew=1,label='net')
        ax.axhline(0,color='#999',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in d.param_value])
        title(ax,f'{variant}SPIN privacy flips'); ax.set_xlabel(xl); ax.set_ylabel('# items')
    axs[0].legend(fontsize=7)
    fig.suptitle('Flip decomposition (Qwen2-7B privacy) — gains are genuine corrections',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.85); return fig
reg('X04_flip_decomp',fflip)

# ---- X05 iterSPIN compute cost ----
def fcost():
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    for m in ALL4:
        d=g(model=m,metric='round_seconds')[['param_value','value']].dropna().drop_duplicates('param_value').sort_values('param_value')
        if not d.empty:
            cum=d.value.cumsum()/60.0
            ax.plot(d.param_value,cum,'-o',color=MODc[m],ms=5,mec='white',mew=1,lw=1.8,label=MODn[m])
    title(ax,'iterSPIN cumulative identification cost'); ax.set_xlabel('rounds n'); ax.set_ylabel('cumulative minutes (2×T4)'); ax.legend(fontsize=7); return fig
reg('X05_compute_cost',fcost)

# ---- X06 HSIC estimator agreement ----
def festim():
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    VC={'soft':'#D06A88','iter':'#7B6FC7','baseline':'#2BA88C'}
    for v,c in VC.items():
        r=g(variant=v,domain='clinical',metric='hsic')[['arm','param_value','sigma','delta']].dropna()
        nn=g(variant=v,domain='clinical',metric='hsic_norm')[['arm','param_value','sigma','delta']].dropna()
        j=r.merge(nn,on=['arm','param_value','sigma'],suffixes=('_raw','_norm'))
        if not j.empty: axs[0].scatter(j.delta_raw,j.delta_norm,color=c,s=28,ec='white',lw=.6,alpha=.85,label={'baseline':'hard SPIN','soft':'softSPIN','iter':'iterSPIN'}[v])
    lim=[-10,2]; axs[0].plot(lim,lim,'--',color='#AAA',lw=1); axs[0].axhline(0,color='#DDD',lw=.6); axs[0].axvline(0,color='#DDD',lw=.6)
    title(axs[0],'raw vs normalised ΔHSIC (%) — clinical'); axs[0].set_xlabel('raw ΔHSIC %'); axs[0].set_ylabel('normalised ΔHSIC %'); axs[0].legend(fontsize=7)
    for v,c in VC.items():
        r=g(variant=v,domain='clinical',metric='hsic')[['arm','param_value','sigma','delta']].dropna()
        ck=g(variant=v,domain='clinical',metric='cka')[['arm','param_value','sigma','delta']].dropna()
        j=r.merge(ck,on=['arm','param_value','sigma'],suffixes=('_h','_c'))
        if not j.empty: axs[1].scatter(j.delta_h,j.delta_c,color=c,s=28,ec='white',lw=.6,alpha=.85)
    axs[1].axhline(0,color='#DDD',lw=.6); axs[1].axvline(0,color='#DDD',lw=.6)
    title(axs[1],'ΔHSIC vs ΔCKA (%) — clinical'); axs[1].set_xlabel('ΔHSIC %'); axs[1].set_ylabel('ΔCKA %')
    fig.suptitle('Do the representational metrics agree? (raw/normalised/CKA diverge)',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.85); return fig
reg('X06_estimator_agreement',festim)

# ---- X07 significance heatmap ----
def fheat():
    from matplotlib.colors import LinearSegmentedColormap
    cm=LinearSegmentedColormap.from_list('tr',['#D06A88','#F3D9E1','#FCFCFB','#CDEBE1','#2BA88C'])
    rows=[]
    for variant,arm,pref in [('soft','soft','α='),('iter','iter','n=')]:
        for axis in ['fairness','privacy']:
            d=g(variant=variant,domain='general',model='Qwen2-7B-Instruct',arm=arm,metric='safe_rate',axis=axis)[['param_value','delta','p']].dropna().drop_duplicates('param_value').sort_values('param_value')
            for _,r in d.iterrows(): rows.append((f'{pref}{r.param_value:g}',axis,r.delta*100,r.p))
    import pandas as pd
    df=pd.DataFrame(rows,columns=['cfg','axis','d','p'])
    cfgs=list(dict.fromkeys(df.cfg)); axes=['fairness','privacy']
    Mz=np.full((len(cfgs),2),np.nan)
    for i,c in enumerate(cfgs):
        for j,a in enumerate(axes):
            v=df[(df.cfg==c)&(df.axis==a)]
            if len(v): Mz[i,j]=v.d.iloc[0]
    fig,ax=plt.subplots(figsize=(4.6,6.5))
    im=ax.pcolormesh(Mz.T,cmap=cm,vmin=-12,vmax=12,edgecolors='white',lw=1)
    for i,c in enumerate(cfgs):
        for j,a in enumerate(axes):
            v=df[(df.cfg==c)&(df.axis==a)]
            if len(v):
                txt=f'{v.d.iloc[0]:+.1f}'+('*' if v.p.iloc[0]<0.05 else '')
                ax.text(i+.5,j+.5,txt,ha='center',va='center',fontsize=7)
    ax.set_xticks(np.arange(len(cfgs))+.5); ax.set_xticklabels(cfgs,rotation=90,fontsize=7)
    ax.set_yticks([.5,1.5]); ax.set_yticklabels(axes); title(ax,'Δ safe rate (pts) — * = p<0.05')
    fig.colorbar(im,ax=ax,shrink=.6,label='Δ pts'); fig.suptitle('Significance overview (Qwen2-7B)',x=.05,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.93,bottom=.2); return fig
reg('X07_sig_heatmap',fheat)

# ---- X08 sparsity vs effect ----
def fspars():
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    for m in ALL4:
        z=g(variant='iter',domain='general',model=m,arm='iter',metric='cumulative_zeroed')[['param_value','value']].dropna().drop_duplicates('param_value')
        a=g(variant='iter',domain='general',model=m,arm='iter',metric='safe_rate',axis='privacy')[['param_value','delta']].dropna().drop_duplicates('param_value')
        j=z.merge(a,on='param_value').sort_values('value')
        if not j.empty:
            lw=2.4 if m=='Qwen2.5-3B-Instruct' else 1.6
            ax.plot(j.value,j.delta*100,'-o',color=MODc[m],ms=5,mec='white',mew=1,lw=lw,label=MODn[m])
    ax.axhline(0,color='#CCC',lw=.8); title(ax,'budget vs effect — iterSPIN (more weights ≠ more decoupling)')
    ax.set_xlabel('cumulative weights zeroed'); ax.set_ylabel('Δ privacy safe rate (pts)'); ax.legend(fontsize=7); return fig
reg('X08_sparsity_effect',fspars)

# ---- X09 anchor / route reproducibility (from verified anchor table) ----
def fanchor():
    routes=['Regular\nSPIN','softSPIN\nα=0','iterSPIN\nn=1','clinical']
    weights=[1149,1148,1201,1163]; fair=[5.54,5.27,6.05,np.nan]; priv=[6.85,6.54,7.61,np.nan]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(4); axs[0].bar(x,weights,color=['#7B6FC7','#7B6FC7','#E0913D','#2BA88C'],ec='white')
    for i,w in enumerate(weights): axs[0].text(i,w+8,str(w),ha='center',fontsize=8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(routes,fontsize=8); title(axs[0],'coupled weights identified'); axs[0].set_ylabel('weights'); axs[0].set_ylim(1100,1230)
    w=.36; x3=np.arange(3)
    axs[1].bar(x3-w/2,fair[:3],w,color='#2BA88C',ec='white',label='fairness'); axs[1].bar(x3+w/2,priv[:3],w,color='#D06A88',ec='white',label='privacy')
    axs[1].set_xticks(x3); axs[1].set_xticklabels([r.replace('\n',' ') for r in routes[:3]],fontsize=8); title(axs[1],'behaviour agrees across routes (Δ pts)'); axs[1].legend(fontsize=7)
    fig.suptitle('Hard-suppression anchor — reproducible across pipelines',x=.07,ha='left',fontsize=12,weight='bold'); fig.subplots_adjust(top=.85)
    axs[1].text(.5,-.28,'Source: Table~anchor (verified)',transform=axs[1].transAxes,ha='center',fontsize=7,color='#888'); return fig
reg('X09_anchor_routes',fanchor)





print("\nGENERATED:",len(figs),"figures")
# ---- contact sheet ----
import math
from PIL import Image
imgs=[f'fig_out/{n}.png' for n in figs]
cols=3; rows=math.ceil(len(imgs)/cols)
thumbs=[Image.open(p) for p in imgs]
w=max(t.width for t in thumbs); h=max(t.height for t in thumbs)
sheet=Image.new('RGB',(cols*w,rows*h),'white')
for i,t in enumerate(thumbs):
    sheet.paste(t,((i%cols)*w,(i//cols)*h))
sheet.save('/mnt/user-data/outputs/ALL_FIGURES_contact_sheet.png')
print("contact sheet saved")
