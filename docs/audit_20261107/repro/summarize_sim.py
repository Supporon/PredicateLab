"""Summarise JSONL calibration output with Wilson intervals (no SciPy needed)."""
import glob,json,math,sys

def wilson(k,n,z=1.959963984540054):
 if n==0:return (None,None)
 p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; r=z/d*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return c-r,c+r
for fn in sorted(glob.glob(sys.argv[1])):
 rs=[json.loads(x) for x in open(fn) if x.strip()]
 n=len(rs); err=sum('error' in r for r in rs)
 dc=sum(r.get('n_validated',0)>0 for r in rs)
 cc=sum(r.get('n_conf_confirmed',0)>0 for r in rs)
 fc=sum(r.get('final_state')=='independently_confirmed' for r in rs)
 print(json.dumps({'file':fn.split('/')[-1],'n':n,'errors':err,
   'discovery_any':{'k':dc,'rate':dc/n if n else None,'wilson95':wilson(dc,n)},
   'confirmation_any':{'k':cc,'rate':cc/n if n else None,'wilson95':wilson(cc,n)},
   'final_positive':{'k':fc,'rate':fc/n if n else None,'wilson95':wilson(fc,n)},
   'mean_validated':sum(r.get('n_validated',0) for r in rs)/n if n else None,
   'mean_conf_confirmed':sum(r.get('n_conf_confirmed',0) for r in rs)/n if n else None,
   'total_validated':sum(r.get('n_validated',0) for r in rs),
   'true_family_validated':sum(r.get('validated_true_family',0) for r in rs),
   'total_conf_confirmed':sum(r.get('n_conf_confirmed',0) for r in rs),
   'true_family_conf_confirmed':sum(r.get('conf_confirmed_true_family',0) for r in rs),
   'true_family_final_positive':sum(1 for r in rs if r.get('final_state')=='independently_confirmed' and r.get('final_true_family')),
   'round0_any_validated':sum(r.get('n_validated_by_round',{}).get('0',0)>0 for r in rs),
   'round1_any_validated':sum(r.get('n_validated_by_round',{}).get('1',0)>0 for r in rs),
  },indent=2))
