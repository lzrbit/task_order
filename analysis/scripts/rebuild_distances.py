#!/usr/bin/env python3
"""Reconstruct five candidate distances from numeric trajectory members.

All candidate parameters are read from inputs/distance_parameters.json. The
reconstruction never examines algorithm performance. Large derived arrays stay
in memory; only compact matrices, counts, hashes, and verification are saved.
"""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

BASE=Path(__file__).resolve().parents[1]

def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def rebuild(root,base=BASE):
 root=Path(root);out=base/'inputs';params=json.loads((out/'distance_parameters.json').read_text())
 groups={};checks=[];sources=[];counts=[]
 for group,names in params['groups'].items():
  parts={'abs_speed':[],'acceleration':[]}
  for name in names:
   path=root/f'interaction_dataset/train_{name}.npz';ss=[];aa=[]
   with zipfile.ZipFile(path) as z:
    with z.open('trajectory.npy') as f:
     ver=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,ver)
     assert not dtype.hasobject and not fort
     assert shape[2]==params['history_timesteps']
     elems=int(np.prod(shape[1:]));rowbytes=elems*dtype.itemsize
     for first in range(0,shape[0],512):
      n=min(512,shape[0]-first);data=f.read(n*rowbytes)
      assert len(data)==n*rowbytes
      traj=np.frombuffer(data,dtype=dtype).reshape((n,)+shape[1:]);ego=traj[:,params['ego_agent_index'],:,:]
      vx,vy=params['velocity_columns'];speed=np.sqrt(ego[:,:,vx]**2+ego[:,:,vy]**2)
      acc=np.diff(speed,axis=1)/params['dt_seconds'];lo,hi=params['acceleration_filter_inclusive']
      ss.append(speed.ravel().copy());aa.append(acc[(acc>=lo)&(acc<=hi)].copy())
   calculated={'abs_speed':np.concatenate(ss),'acceleration':np.concatenate(aa)}
   cache_path=root/f'outputs/train_{name}_combined_features.npz'
   with np.load(cache_path,allow_pickle=False) as cache:
    for channel,a in calculated.items():
     assert np.isfinite(a).all()
     c=cache[channel];same=np.array_equal(a,c)
     checks.append({'dataset':name,'scenario':group,'channel':channel,'n_trajectories':shape[0],'n_features':len(a),'cached_n_features':len(c),'bitwise_equal':same,'max_abs_diff':float(np.max(np.abs(a-c))) if len(a)==len(c) else None,'derived_array_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
     parts[channel].append(a)
   sources.append({'source':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':file_hash(path),'cached_features':str(cache_path.relative_to(root)),'cache_sha256':file_hash(cache_path)})
  groups[group]={k:np.concatenate(v) for k,v in parts.items()}
  counts.append({'scenario':group,'trajectory_rows':sum(x['n_trajectories'] for x in checks if x['scenario']==group and x['channel']=='abs_speed'),'abs_speed_observations':len(groups[group]['abs_speed']),'acceleration_observations':len(groups[group]['acceleration'])})
 scenarios=params['canonical_scenarios'];matrices={}
 for feature in ['abs_speed','acceleration']:
  M=np.zeros((4,4))
  for i in range(4):
   for j in range(i+1,4):M[i,j]=M[j,i]=wasserstein_distance(groups[scenarios[i]][feature],groups[scenarios[j]][feature])
  matrices[feature+'__wasserstein_distance']=M
 speed=[groups[s]['abs_speed'] for s in scenarios];pool=np.concatenate(speed)
 edges=np.linspace(pool.min(),pool.max(),params['histogram_bins']+1);hist=[]
 for a in speed:
  counts_=np.histogram(a,bins=edges)[0];pr=counts_/counts_.sum();pr+=params['histogram_probability_epsilon_per_bin'];pr/=pr.sum();hist.append(pr)
 KL=np.zeros((4,4));JS=np.zeros((4,4))
 for i in range(4):
  for j in range(4):
   KL[i,j]=np.sum(hist[i]*np.log(hist[i]/hist[j]));JS[i,j]=jensenshannon(hist[i],hist[j])**2
 matrices['abs_speed__kl_divergence']=KL;matrices['abs_speed__jensen-shannon_divergence']=JS
 mu=pool.mean();sd=pool.std(ddof=0);rng=np.random.default_rng(params['mmd_seed']);samples=[];sample_indices=[]
 for a in speed:
  idx=rng.choice(len(a),min(params['mmd_sample_per_task'],len(a)),replace=False)
  sample_indices.append(idx);samples.append((a[idx]-mu)/sd)
 def kernel_mean(x,y):
  total=0.
  for k in range(0,len(x),256):total+=np.exp(-params['mmd_gamma']*(x[k:k+256,None]-y[None,:])**2).sum()
  return total/(len(x)*len(y))
 selfs=[kernel_mean(x,x) for x in samples];M=np.zeros((4,4))
 for i in range(4):
  for j in range(i+1,4):M[i,j]=M[j,i]=np.sqrt(max(0.,selfs[i]+selfs[j]-2*kernel_mean(samples[i],samples[j])))
 matrices['abs_speed__mmd']=M
 legacy=pd.read_csv(out/'legacy_distance_matrices.csv').set_index(['distance','from','to'])
 rows=[]
 for key,M in matrices.items():
  for i in range(4):
   for j in range(4):
    old=legacy.loc[(key,scenarios[i],scenarios[j])]
    rows.append({'distance':key,'from':scenarios[i],'to':scenarios[j],'legacy':old.legacy,'symmetrized':old.symmetrized,'corrected':M[i,j],'exact_full_sample':M[i,j] if key.endswith('wasserstein_distance') else np.nan})
 pd.DataFrame(rows).to_csv(out/'distance_matrices.csv',index=False)
 pd.DataFrame(checks).to_csv(out/'raw_feature_verification.csv',index=False)
 pd.DataFrame(sources).to_csv(out/'trajectory_source_sha256.csv',index=False)
 pd.DataFrame(counts).to_csv(out/'feature_counts.csv',index=False)
 # Store indices (8,000 integers), not the large source arrays, for exact audit.
 pd.DataFrame([{'scenario':scenarios[i],'draw':j,'row_index':int(v)} for i,idx in enumerate(sample_indices) for j,v in enumerate(idx)]).to_csv(out/'mmd_sample_indices.csv',index=False)
 metadata={'all_24_dataset_channel_checks_bitwise_equal':all(c['bitwise_equal'] for c in checks),'histogram_edges':edges.tolist(),'pooled_speed_mean':float(mu),'pooled_speed_sd':float(sd),'W1_speed_units':'m/s, inherited from source velocity convention','W1_acceleration_units':'m/s^2, derived with dt=0.1 s','KL_JS_units':'dimensionless, natural logarithms','MMD_units':'dimensionless, pooled z-scored speed','source_scope':'These archived trajectory arrays reproduce the stored feature caches exactly and supply the post-training distance analysis. Model training is maintained separately.','numpy_version':np.__version__}
 (out/'distance_reconstruction.json').write_text(json.dumps(metadata,indent=2))
 print('Rebuilt five candidates from raw numeric trajectories; cache matches:',metadata['all_24_dataset_channel_checks_bitwise_equal'])
 return matrices

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True,type=Path);args=parser.parse_args();rebuild(args.archive)
