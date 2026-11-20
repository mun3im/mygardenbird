import csv,re,sys,numpy as np
from math import sqrt
B='/Users/mun3im/Dropbox/Conda/MyGardenBird/birdnet_results/'
rng=np.random.default_rng(42)
def wilson(k,n,z=1.96):
    p=k/n; c=(p+z*z/2/n)/(1+z*z/n); h=z*sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return 100*(c-h),100*(c+h)
for rate in ('16khz','44khz'):
    rows=list(csv.DictReader(open(B+f'final15b_{rate}/per_clip_results.csv')))
    src=[re.search(r'xc(\d+)_',r['filepath']).group(1) for r in rows]
    cls=np.array([r['ground_truth'] for r in rows]); y=np.array([int(float(r['correct'])) if r['correct'] not in('True','False') else r['correct']=='True' for r in rows]).astype(int)
    ids={s:i for i,s in enumerate(sorted(set(src)))}; sid=np.array([ids[s] for s in src]); S=len(ids)
    print(rate,'n',len(y),'sources',S,'acc',round(100*y.mean(),2),'wilson',[round(v,2) for v in wilson(y.sum(),len(y))])
    def boot(mask,nb=10000):
        yy=y[mask]; ss=sid[mask]; u=np.unique(ss); m={s:i for i,s in enumerate(u)}; ss=np.array([m[s] for s in ss])
        k=np.bincount(ss,weights=yy,minlength=len(u)); n=np.bincount(ss,minlength=len(u))
        idx=rng.integers(0,len(u),(nb,len(u)))
        a=100*k[idx].sum(1)/n[idx].sum(1); return np.percentile(a,[2.5,97.5]),len(u)
    ci,_=boot(np.ones(len(y),bool)); print(' overall source-bootstrap',np.round(ci,2))
    for c in sorted(set(cls)):
        mk=cls==c; ci,ns=boot(mk); w=wilson(y[mk].sum(),mk.sum())
        print(f'  {c:28s} n={mk.sum():4d} src={ns:3d} acc={100*y[mk].mean():5.1f} boot=[{ci[0]:.1f},{ci[1]:.1f}] wilson=[{w[0]:.1f},{w[1]:.1f}]')
