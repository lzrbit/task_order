#!/usr/bin/env python3
"""Rebuild frozen inputs from the local experiment archive, without pickle.

Usage: python extract_inputs.py --archive "/path/to/NTU-CL Task Order"
All five primary distances are rebuilt from numerical trajectories using frozen parameters.
NPZ object arrays are never read. Source files are read-only.
"""
from pathlib import Path
import argparse,hashlib,re,json,zipfile
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);args=p.parse_args()
root=args.archive.resolve();base=Path(__file__).resolve().parents[1];out=base/'inputs';out.mkdir(exist_ok=True)
scenarios=['urban_merge','highway_merge','intersection','roundabout']
legacy_order=['highway_merge','intersection','roundabout','urban_merge'];perm=[legacy_order.index(s) for s in scenarios]
pat=re.compile(r'Trained after task(\d+)--> Tested in task:(\d+).*?minFDE:\s*([-\d.eE+]+)m\s*MR:\s*([-\d.eE+]+)%',re.S)
rows=[];manifest=[]
for path in sorted(root.glob('exp_results/random_*/order_*/*.txt')):
 seed=int(path.parts[-3].split('_')[-1]);order=int(path.parts[-2].split('_')[-1]);m=re.fullmatch(r'After_(\d+)_CL_tasks_(.*?)_bf_2000.txt',path.name)
 if not m:continue
 stage=int(m[1]);algo=m[2];records=pat.findall(path.read_text())
 if len(records)!=4:raise ValueError(f'Expected four test-task records: {path}')
 for st,task,fde,mr in records:
  if int(st)!=stage:raise ValueError(f'Stage mismatch: {path}')
  rows.append({'seed':seed,'Order':order,'algorithm':algo,'stage':int(st),'task':int(task),'FDE':float(fde),'MR':float(mr)})
 manifest.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
raw=pd.DataFrame(rows).sort_values(['seed','Order','algorithm','stage','task'])
assert len(raw)==4608 and not raw.duplicated(['seed','Order','algorithm','stage','task']).any()
raw.to_csv(out/'stage_task_results.csv',index=False);pd.DataFrame(manifest).to_csv(out/'raw_log_sha256.csv',index=False)
metrics=[]
for (seed,order,algo),g in raw.groupby(['seed','Order','algorithm']):
 for metric in ['FDE','MR']:
  R=g.pivot(index='stage',columns='task',values=metric).values
  values={'CT':np.diag(R).mean(),'BWT':np.mean([np.mean([R[r,i]-R[i,i] for i in range(r)]) for r in range(1,4)])}
  for kind,value in values.items():metrics.append({'seed':seed,'Order':order,'algorithm':algo,'metric':metric+'-'+kind,'value':value})
pd.DataFrame(metrics).to_csv(out/'performance_per_seed.csv',index=False)
orders=[]
for line in (root/'exp_results/task_order_info.txt').read_text().splitlines():
 m=re.search(r'(\d+):\s*\{(.+)\}',line)
 if m:
  seq=[x.strip() for x in m[2].split(',')]
  orders.append({'Order':int(m[1]),'Sequence':' → '.join('ABCD'[scenarios.index(x)] for x in seq),'scenario_sequence':'|'.join(seq),'first_task':seq[0]})
pd.DataFrame(orders).to_csv(out/'orders.csv',index=False)
legacy_path=root/'similarity_analysis/export_single/distance_matrices_normalized.npz'
with np.load(legacy_path,allow_pickle=False) as archive:legacy={k:archive[k][np.ix_(perm,perm)] for k in archive.files}
rows=[]
for key,M in legacy.items():
 repaired=M if 'kl_divergence' in key else (M+M.T)/2
 for i in range(4):
  for j in range(4):rows.append({'distance':key,'from':scenarios[i],'to':scenarios[j],'legacy':M[i,j],'symmetrized':(M[i,j]+M[j,i])/2,'corrected':repaired[i,j]})
pd.DataFrame(rows).to_csv(out/'legacy_distance_matrices.csv',index=False)
from rebuild_distances import rebuild
rebuild(root,base)
# Header-only data provenance; object-valued members are intentionally skipped.
datasets=[]
for path in sorted((root/'interaction_dataset').glob('*.npz')):
 with zipfile.ZipFile(path) as z:
  with z.open('trajectory.npy') as f:
   version=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,version)
 datasets.append({'file':path.name,'trajectory_rows':shape[0],'trajectory_shape':'x'.join(map(str,shape)),'dtype':str(dtype)})
pd.DataFrame(datasets).to_csv(out/'dataset_sizes.csv',index=False)
provenance={'source':'exp_results/random_{0,1,2}/order_{0..23}/After_{1..4}_CL_tasks_{vanilla,der,syrem,gsm}_bf_2000.txt','log_count':len(manifest),'primary_distance_source':'Numeric trajectory.npy members of twelve archived NPZ datasets; fully specified by distance_parameters.json and rebuild_distances.py. All speed/acceleration observations verified bitwise against the feature caches.','source_legacy_distance_file':str(legacy_path.relative_to(root)),'source_legacy_distance_sha256':hashlib.sha256(legacy_path.read_bytes()).hexdigest(),'legacy_scenario_order':legacy_order,'canonical_scenario_order':scenarios,'legacy_note':'Original directed matrices and symmetric repairs retained only as sensitivity inputs. Legacy feature sampling, RNG state and scaling cannot be recovered. This does not affect the newly reconstructed primary distances.','training_note':'Training and its configuration are maintained separately. This repository covers evaluation outcomes and post-training analysis. Dataset row counts describe the archived feature-analysis inputs. Run directory indices label training replicates and do not assert paired random seeds.'}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
# Compare independently parsed metrics with the historical rounded summary CSVs.
perf=pd.DataFrame(metrics);checks=[]
for metric in perf.metric.unique():
 for seed in range(3):
  derived=perf[(perf.metric==metric)&(perf.seed==seed)].pivot(index='Order',columns='algorithm',values='value')
  stored=pd.read_csv(root/f'exp_results/performance/summary_random_{seed}_{metric}.csv').set_index('Order')
  stored.columns=[c.split('_',1)[1] for c in stored.columns];stored=stored[derived.columns]
  checks.append({'metric':metric,'seed':seed,'max_abs_difference':float(abs(derived-stored).values.max())})
 derived=perf[perf.metric==metric].groupby(['Order','algorithm']).value.mean().unstack()
 stored=pd.read_csv(root/f'exp_results/performance/summary_average_{metric}.csv').set_index('Order');stored.columns=[c.split('_',1)[1] for c in stored.columns];stored=stored[derived.columns]
 checks.append({'metric':metric,'seed':'average','max_abs_difference':float(abs(derived-stored).values.max())})
assert max(x['max_abs_difference'] for x in checks)<8e-7
(base/'results').mkdir(exist_ok=True)
(base/'results/source_validation.json').write_text(json.dumps({'invalid_logs':[],'checks':checks},indent=2))
print('Extracted inputs to',out)
