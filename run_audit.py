#!/usr/bin/env python3
"""Reproduces every claim in DECISIONS.md. Usage: python audit/run_audit.py --data data"""
import argparse, numpy as np, pandas as pd, os
ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); a=ap.parse_args()
D=a.data; dc=[f'd_{i}' for i in range(1,1914)]
sales=pd.read_csv(f'{D}/sales_train.csv'); cal=pd.read_csv(f'{D}/calendar.csv')
pr=pd.read_csv(f'{D}/sell_prices.csv'); mkt=pd.read_csv(f'{D}/market_signal.csv')
ven=pd.read_csv(f'{D}/vendor_signal.csv')
X=sales[dc].to_numpy(float); ids=sales['id'].tolist(); item=sales['item_id'].to_numpy()
for f in (mkt,ven): f['dn']=f['d'].str.replace('d_','').astype(int)

print("[1] market_signal leak")
print("    coverage d_%d..d_%d (horizon rows: %d)"%(mkt['dn'].min(),mkt['dn'].max(),(mkt['dn']>1913).sum()))
lg=sales.melt(id_vars=['id'],value_vars=dc,var_name='d',value_name='u'); lg['dn']=lg['d'].str.replace('d_','').astype(int)
m=lg.merge(mkt,on=['id','dn'])
zu=m['u']==0; zs=m['mkt_signal']==0
print("    units==0: %d | mkt==0: %d | perfect alignment: %s"%(zu.sum(),zs.sum(),(zu==zs).all()))
c=[np.corrcoef(g['mkt_signal'],g['u'])[0,1] for _,g in m.groupby('id')]
print("    mean per-series lag-0 corr: %.4f"%np.mean(c))

print("[2] vendor_signal provenance")
vh=ven[ven['dn']>1913].groupby('id')['vendor_forecast'].mean().reindex(ids)
print("    coverage d_%d..d_%d | corr(horizon, full-history mean)=%.4f"%(ven['dn'].min(),ven['dn'].max(),np.corrcoef(vh,X.mean(1))[0,1]))
k3=[i for i,s in enumerate(ids) if 'KA_3' in s]
print("    KA_3 over-forecast: %.2fx"%(vh.iloc[k3].sum()/X[k3,1851:].mean(1).sum()))

print("[3] KA_3 break")
tot=X[k3].sum(0); best=max(((cp,abs(tot[1600:cp].mean()-tot[cp:].mean())/np.sqrt(tot[1600:cp].var()/(cp-1600)+tot[cp:].var()/(1913-cp))) for cp in range(1700,1880)),key=lambda r:r[1])
print("    changepoint d_%d t=%.1f  %.1f -> %.1f (%.2fx)"%(best[0]+1,best[1],tot[1600:best[0]].mean(),tot[best[0]:].mean(),tot[best[0]:].mean()/tot[1600:best[0]].mean()))
for st in ['KA_1','KA_2','TN_1','MH_1']:
    g=sales[sales['store_id']==st][dc].to_numpy(float).sum(0)
    print("    control %s ratio %.3f"%(st,g[1851:].mean()/g[1600:1851].mean()))
i=ids.index('GROCERY_3_ATTA_KA_3_validation')
print("    ATTA_KA_3 zero-days post-break: %.1f%% (stockout would be high)"%((X[i,1851:]==0).mean()*100))

print("[4] CABLE dead window")
cb=[i for i,s in enumerate(ids) if 'CABLE' in s]
print("    inside d_961-1440 %.3f | outside %.3f | post-recovery d_1441+ %.3f | pre-dip d_1-960 %.3f"%(
  X[cb][:,960:1440].mean(),np.r_[X[cb][:,:960].ravel(),X[cb][:,1440:].ravel()].mean(),X[cb][:,1440:].mean(),X[cb][:,:960].mean()))
wk=set(cal[cal['d'].isin([f'd_{i}' for i in range(961,1441)])]['wm_yr_wk'])
pc=pr[pr['item_id']=='ELECTRONICS_1_CABLE']
print("    priced weeks inside window: %d of %d"%(len(wk&set(pc['wm_yr_wk'])),len(wk)))

print("[5] MH_2 PICKLE promo")
pp=pr[(pr['item_id']=='GROCERY_3_PICKLE')&(pr['store_id']=='MH_2')]
print("    weeks priced <2.0: %s"%pp[pp['sell_price']<2.0]['wm_yr_wk'].tolist())
i=ids.index('GROCERY_3_PICKLE_MH_2_validation')
print("    promo wk2040 (d_997-1003) %.2f/day | 4wk before %.2f | 4wk after %.2f | series mean %.2f"%(
  X[i,996:1003].mean(),X[i,968:996].mean(),X[i,1003:1031].mean(),X[i].mean()))

print("[6] snap has no signal")
h=cal[cal['d'].isin(dc)].reset_index(drop=True)
for st in ['MH','KA','TN']:
    xs=sales[sales['store_id'].str.startswith(st)][dc].to_numpy(float).sum(0)
    f=h[f'snap_{st}'].to_numpy(); m2=np.array([int(d.split('_')[1])>1200 for d in h['d']])
    print("    %s ratio %.3f"%(st,xs[(f==1)&m2].mean()/xs[(f==0)&m2].mean()))

print("[7] metric structure")
v=X[:,-90:].sum(1); w=v/v.sum()
print("    ATTA share of volume (drives WAPE): %.1f%%"%(w[[i for i,s in enumerate(ids) if 'ATTA' in s]].sum()*100))
print("    series >70%% zero-days (drive mean RMSSE): %d of 60"%((X==0).mean(1)>0.7).sum())