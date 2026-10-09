#!/usr/bin/env python3
"""Recompute the descriptive paper analysis from frozen CSV inputs.

No model training, pickle loading, or inferential p-values. The only stochastic
step is a replicate-level bootstrap (fixed seed, cell-wise resampling of the three
training replicates) that reports leader probabilities and winner-count intervals.
Primary distances are reconstructed from numeric trajectories using the frozen
distance_parameters.json. KL remains directional; the other matrices are symmetric.
D=(last transition-first transition)/2 is evaluated algebraically, retaining exact
ties. Scenario order is A,B,C,D as below.
"""
from pathlib import Path
import json
from itertools import permutations, combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

BASE=Path(__file__).resolve().parents[1]
IN=BASE/'inputs';OUT=BASE/'results';OUT.mkdir(exist_ok=True)
SCENARIOS=['urban_merge','highway_merge','intersection','roundabout']
ALGOS=['vanilla','der','syrem','gsm']
LABEL={'vanilla':'Vanilla','der':'DER','syrem':'SyReM','gsm':'GSM'}
METRICS=['FDE-CT','MR-CT','FDE-BWT','MR-BWT']
CANDIDATES=[('W1 speed','abs_speed__wasserstein_distance'),('KL speed','abs_speed__kl_divergence'),('JS speed','abs_speed__jensen-shannon_divergence'),('MMD speed','abs_speed__mmd'),('W1 acceleration','acceleration__wasserstein_distance')]
BOOTSTRAP_DRAWS=5000;BOOTSTRAP_SEED=0
orders=pd.read_csv(IN/'orders.csv').sort_values('Order')
seqs=[[SCENARIOS.index(s) for s in v.split('|')] for v in orders.scenario_sequence]
seq_letters=[''.join('ABCD'[i] for i in s) for s in seqs]
first=np.array([s[0] for s in seqs]);no_A=first!=0
posA=np.array([s.index(0)+1 for s in seqs])  # presentation position (1-4) of task A
perf=pd.read_csv(IN/'performance_per_seed.csv')
raw=pd.read_csv(IN/'stage_task_results.csv')
dist=pd.read_csv(IN/'distance_matrices.csv')
def matrix(key,col='corrected'):
 return dist[dist.distance==key].pivot(index='from',columns='to',values=col).reindex(index=SCENARIOS,columns=SCENARIOS).values
def descriptor(M):return np.array([(M[o[2],o[3]]-M[o[0],o[1]])/2 for o in seqs])
def rho(x,y):return float(spearmanr(x,y).statistic)
def group_residual(x,rank=False,groups=None):
 g=first if groups is None else groups
 v=rankdata(x) if rank else np.asarray(x,dtype=float)
 return v-np.array([v[g==k].mean() for k in g])
def adjusted_rho(x,y,groups=None):return float(pearsonr(group_residual(x,True,groups),group_residual(y,True,groups)).statistic)
def decompose_variation(x):
 """Order, algorithm, interaction, and within-cell SS for a balanced array."""
 n_orders,n_algorithms,n_replicates=x.shape
 grand=x.mean();total=np.sum((x-grand)**2)
 order=x.mean((1,2))-grand;algorithm=x.mean((0,2))-grand
 interaction=x.mean(2)-grand-order[:,None]-algorithm[None,:]
 within=x-x.mean(2)[:,:,None]
 pieces={'order':n_algorithms*n_replicates*np.sum(order**2),'algorithm':n_orders*n_replicates*np.sum(algorithm**2),'order_x_algorithm':n_replicates*np.sum(interaction**2),'within_cell_training':np.sum(within**2)}
 assert np.isclose(sum(pieces.values()),total)
 return total,pieces
def split_order_by_start(x,starts):
 """Split the order SS into a starting-task part and an order-within-start part."""
 n_orders,n_algorithms,n_replicates=x.shape
 grand=x.mean();order_means=x.mean((1,2))
 start_means=np.array([order_means[starts==g].mean() for g in starts])
 ss_start=n_algorithms*n_replicates*np.sum((start_means-grand)**2)
 ss_within_start=n_algorithms*n_replicates*np.sum((order_means-start_means)**2)
 return ss_start,ss_within_start
def cell_array(frame,m):
 x=np.empty((24,4,3))
 for ai,a in enumerate(ALGOS):
  x[:,ai,:]=frame[(frame.metric==m)&(frame.algorithm==a)].pivot(index='Order',columns='seed',values='value').reindex(range(24)).values
 return x
M=matrix(CANDIDATES[0][1]);D=descriptor(M)
assert np.array_equal(M,M.T)
assert len(np.unique(D))==6
assert np.allclose(np.sort(D),-np.sort(D)[::-1])
assert all((D[i]<0)==(posA[i]<=2) for i in range(24))  # sign(D) coincides with "A in positions 1-2"
means=perf.groupby(['Order','metric','algorithm']).value.mean()
stds=perf.groupby(['Order','metric','algorithm']).value.std(ddof=1)
per_order=orders.copy();per_order['D']=D
for k in range(3):per_order[f'Step{k+1}']=[M[s[k],s[k+1]] for s in seqs]
for m in METRICS:
 for a in ALGOS:per_order[m+'__'+a]=means.xs((m,a),level=('metric','algorithm')).reindex(range(24)).values
per_order.to_csv(OUT/'per_order.csv',index=False)
per_order[list(orders.columns)+['D','Step1','Step2','Step3']].to_csv(OUT/'order_descriptor.csv',index=False)
correlations=[]
for label,key in CANDIDATES:
 dc=descriptor(matrix(key)); old=descriptor(matrix(key,'legacy'))
 for m in METRICS:
  for a in ALGOS:
   y=per_order[m+'__'+a].values
   r={'candidate':label,'distance':key,'metric':m,'algorithm':a,'rho_all':rho(dc,y),'rho_without_A':rho(dc[no_A],y[no_A]),'rho_first_task_adjusted':adjusted_rho(dc,y),'rho_legacy':rho(old,y)}
   for g in range(4):r['rho_within_'+str(g)]=rho(dc[first==g],y[first==g])
   correlations.append(r)
corr=pd.DataFrame(correlations);corr.to_csv(OUT/'candidate_correlations.csv',index=False)
corr[corr.candidate=='W1 speed'].to_csv(OUT/'correlations.csv',index=False)
summary=[]
for c,g in corr.groupby('candidate',sort=False):
 rec={'candidate':c,'unique_D_values':int(len(np.unique(descriptor(matrix(dict(CANDIDATES)[c])))))}
 for subset in ['CT','BWT','all']:
  h=g if subset=='all' else g[g.metric.str.endswith(subset)]
  rec['mean_abs_rho_'+subset]=h.rho_all.abs().mean()
 summary.append(rec)
pd.DataFrame(summary).to_csv(OUT/'candidate_summary.csv',index=False)
paths=[]
for g in range(4):
 ids=np.where(first==g)[0];totals=per_order[['Step1','Step2','Step3']].sum(axis=1).values
 for kind,k in [('max',ids[np.argmax(totals[ids])]),('min',ids[np.argmin(totals[ids])])]:
  rec=per_order.loc[k,['Order','Sequence','first_task','Step1','Step2','Step3','D']].to_dict();rec.update(kind=kind,total_distance=totals[k]);paths.append(rec)
path=pd.DataFrame(paths);path.to_csv(OUT/'extreme_paths.csv',index=False)
Dt=(D-D.min())/(D.max()-D.min()) if np.ptp(D)>0 else np.full_like(D,0.5);weighted=[]
assert np.isfinite(Dt).all() and (1-Dt).sum()>0 and Dt.sum()>0
for m in METRICS:
 w=1-Dt if m.endswith('CT') else Dt
 for a in ALGOS:
  y=per_order[m+'__'+a].values
  weighted.append({'metric':m,'algorithm':a,'naive_mean':y.mean(),'D_weighted':np.dot(w,y)/sum(w),'max_diff_path_avg':y[path[path.kind=='max'].Order].mean(),'min_diff_path_avg':y[path[path.kind=='min'].Order].mean(),'effective_order_count':sum(w)**2/sum(w*w)})
pd.DataFrame(weighted).to_csv(OUT/'weighted_metrics.csv',index=False)
# Balanced descriptive decomposition (not ANOVA tests). Replicate identifiers
# are local to each order-algorithm cell, not a matched factor across cells.
variance=[];finite=[];algmeans=[];cells={}
for m in METRICS:
 x=cell_array(perf,m);cells[m]=x
 grand=x.mean();SSt,pieces=decompose_variation(x)
 for term,ss in pieces.items():variance.append({'metric':m,'component':term,'SS':ss,'fraction_total_SS':ss/SSt,'total_SS':SSt})
 y=x.mean(2);wins=np.argmin(y,axis=1)
 finite.append({'metric':m,'grand_mean':grand,'per_order_algorithm_mean_min':y.mean(1).min(),'per_order_algorithm_mean_max':y.mean(1).max(),'per_order_algorithm_mean_range':np.ptp(y.mean(1)),'median_within_order_algorithm_range':np.median(np.ptp(y,axis=1)),'algorithm_mean_gap':np.ptp(y.mean(0))})
 for ai,a in enumerate(ALGOS):algmeans.append({'metric':m,'algorithm':a,'mean':y[:,ai].mean(),'order_min':y[:,ai].min(),'order_max':y[:,ai].max(),'winner_count':int(np.sum(wins==ai))})
pd.DataFrame(variance).to_csv(OUT/'sum_of_squares.csv',index=False)
pd.DataFrame(finite).to_csv(OUT/'finite_order_summary.csv',index=False)
pd.DataFrame(algmeans).to_csv(OUT/'algorithm_summary.csv',index=False)
# Raw-log diagnostics: initial task error, final mean error, BWT numerator components,
# and the warm-start current-task error CT_{2:N} (diagonal of R without the first stage).
extra=[]
for (seed,oid,a),g in raw.groupby(['seed','Order','algorithm']):
 for m in ['FDE','MR']:
  R=g.pivot(index='stage',columns='task',values=m).values
  post=np.mean([R[r,:r].mean() for r in range(1,4)])
  reference=np.mean([np.diag(R)[:r].mean() for r in range(1,4)])
  values={'initial_error':R[0,0],'final_mean_error':R[-1].mean(),'BWT_post_error':post,'BWT_reference_error':reference,'BWT':post-reference,'CT':np.diag(R).mean(),'CT_warm_start':np.diag(R)[1:].mean()}
  for stat,v in values.items():extra.append({'seed':seed,'Order':oid,'algorithm':a,'metric':m,'statistic':stat,'value':v})
extra=pd.DataFrame(extra);extra.to_csv(OUT/'raw_diagnostics_per_seed.csv',index=False)
extras=extra.groupby(['Order','algorithm','metric','statistic']).value.mean().reset_index();extras.to_csv(OUT/'raw_diagnostics_per_order.csv',index=False)
exsumm=[];excor=[]
for (m,stat,a),g in extras.groupby(['metric','statistic','algorithm']):
 y=g.set_index('Order').value.reindex(range(24)).values
 exsumm.append({'metric':m,'statistic':stat,'algorithm':a,'mean':y.mean(),'order_min':y.min(),'order_max':y.max(),'order_range':np.ptp(y),'A_first_mean':y[~no_A].mean(),'non_A_first_mean':y[no_A].mean()})
 excor.append({'metric':m,'statistic':stat,'algorithm':a,'rho_all':rho(D,y),'rho_without_A':rho(D[no_A],y[no_A]),'rho_first_task_adjusted':adjusted_rho(D,y)})
pd.DataFrame(exsumm).to_csv(OUT/'raw_diagnostic_summary.csv',index=False)
pd.DataFrame(excor).to_csv(OUT/'raw_diagnostic_correlations.csv',index=False)
# Warm-start current-task error as a per-seed metric array (same layout as the CT arrays).
warm={}
for m in ['FDE','MR']:
 sub=extra[(extra.metric==m)&(extra.statistic=='CT_warm_start')].rename(columns={'value':'value'})
 sub=sub.assign(metric=m+'-CT2N')
 warm[m+'-CT2N']=cell_array(sub,m+'-CT2N')
 assert np.allclose(warm[m+'-CT2N'],(4*cells[m+'-CT']-cell_array(extra[(extra.metric==m)&(extra.statistic=='initial_error')].assign(metric=m+'-R11'),m+'-R11'))/3)
# Extended decomposition: starting task / order within start / algorithm / interaction / within,
# for the four metrics and the warm-start CT, plus the same decomposition without A-first orders.
extended=[]
for m,x in list(cells.items())+list(warm.items()):
 SSt,pieces=decompose_variation(x);ss_start,ss_within_start=split_order_by_start(x,first)
 assert np.isclose(ss_start+ss_within_start,pieces['order'])
 rec={'metric':m,'subset':'all_orders','n_orders':24,'total_SS':SSt,'start':ss_start/SSt,'order_within_start':ss_within_start/SSt,'order':pieces['order']/SSt,'algorithm':pieces['algorithm']/SSt,'order_x_algorithm':pieces['order_x_algorithm']/SSt,'within_cell_training':pieces['within_cell_training']/SSt}
 extended.append(rec)
 xs=x[no_A];SSt2,p2=decompose_variation(xs);s2,w2=split_order_by_start(xs,first[no_A])
 extended.append({'metric':m,'subset':'excluding_A_first','n_orders':18,'total_SS':SSt2,'start':s2/SSt2,'order_within_start':w2/SSt2,'order':p2['order']/SSt2,'algorithm':p2['algorithm']/SSt2,'order_x_algorithm':p2['order_x_algorithm']/SSt2,'within_cell_training':p2['within_cell_training']/SSt2})
extended=pd.DataFrame(extended);extended.to_csv(OUT/'extended_decomposition.csv',index=False)
# Order-level means by the presentation position of task A, and by starting task, for CT and CT_{2:N}.
position=[]
for m,x in list(cells.items())+list(warm.items()):
 y=x.mean(2).mean(1)
 rec={'metric':m}
 for p in range(1,5):rec[f'posA_{p}_mean']=y[posA==p].mean()
 for g in range(4):rec[f'start_{"ABCD"[g]}_mean']=y[first==g].mean()
 rec['A_first_mean']=y[~no_A].mean();rec['non_A_first_mean']=y[no_A].mean()
 position.append(rec)
pd.DataFrame(position).to_csv(OUT/'position_of_A_means.csv',index=False)
# Descriptor versus position of task A: marginal, position-adjusted, and starting-task-adjusted associations.
poscor=[]
for m,x in list(cells.items())+list(warm.items()):
 y_all=x.mean(2)
 for ai,a in enumerate(ALGOS):
  y=y_all[:,ai]
  poscor.append({'metric':m,'algorithm':a,'rho_D':rho(D,y),'rho_posA':rho(posA,y),'rho_D_without_A_first':rho(D[no_A],y[no_A]),'rho_D_first_task_adjusted':adjusted_rho(D,y),'rho_D_positionA_adjusted':adjusted_rho(D,y,posA),'spearman_D_posA':rho(D,posA)})
poscor=pd.DataFrame(poscor);poscor.to_csv(OUT/'position_correlations.csv',index=False)
# Replicate-level bootstrap: resample the three replicates within every order-algorithm cell,
# recompute order means, uniform and D-weighted scores, per-order leaders and winner counts.
rng=np.random.default_rng(BOOTSTRAP_SEED);boot=[];sep=[]
final_cells={m:cell_array(extra[(extra.metric==m)&(extra.statistic=='final_mean_error')].assign(metric=m+'-Final'),m+'-Final') for m in ['FDE','MR']}
for m,x in list(cells.items())+[('FDE-Final',final_cells['FDE']),('MR-Final',final_cells['MR'])]:
 w=(1-Dt if m.endswith('CT') else Dt) if m in METRICS else None
 ymean=x.mean(2);ysd=x.std(2,ddof=1)
 # Separable per-order leader: gap to the runner-up exceeds the pooled replicate SD of the two.
 separable=[]
 for o in range(24):
  srt=np.argsort(ymean[o]);lead,run=srt[0],srt[1];gap=ymean[o,run]-ymean[o,lead]
  pooled=np.sqrt((ysd[o,lead]**2+ysd[o,run]**2)/2);separable.append(bool(gap>pooled));sep.append({'metric':m,'Order':o,'Sequence':seq_letters[o],'leader':ALGOS[lead],'runner_up':ALGOS[run],'gap':gap,'pooled_sd':pooled,'gap_over_sd':gap/pooled if pooled>0 else np.inf,'separable':bool(gap>pooled)})
 uni_wins=np.zeros(4);wt_wins=np.zeros(4);ex_wins=np.zeros(4);counts=np.zeros((BOOTSTRAP_DRAWS,4))
 for b in range(BOOTSTRAP_DRAWS):
  idx=rng.integers(0,3,size=x.shape);res=np.take_along_axis(x,idx,axis=2).mean(2)
  uni_wins[np.argmin(res.mean(0))]+=1
  ex_wins[np.argmin(res[no_A].mean(0))]+=1
  if w is not None:wt_wins[np.argmin((w[:,None]*res).sum(0)/w.sum())]+=1
  counts[b]=np.bincount(np.argmin(res,axis=1),minlength=4)
 for ai,a in enumerate(ALGOS):
  boot.append({'metric':m,'algorithm':a,'uniform_mean':ymean[:,ai].mean(),'p_best_uniform':uni_wins[ai]/BOOTSTRAP_DRAWS,'p_best_D_weighted':(wt_wins[ai]/BOOTSTRAP_DRAWS) if w is not None else np.nan,'uniform_mean_excl_A_first':ymean[no_A,ai].mean(),'p_best_uniform_excl_A_first':ex_wins[ai]/BOOTSTRAP_DRAWS,'winner_count':int(np.sum(np.argmin(ymean,axis=1)==ai)),'winner_count_q025':float(np.percentile(counts[:,ai],2.5)),'winner_count_q975':float(np.percentile(counts[:,ai],97.5)),'separable_leader_orders':int(sum(1 for o in range(24) if separable[o] and np.argmin(ymean[o])==ai)),'separable_orders_total':int(sum(separable)),'bootstrap_draws':BOOTSTRAP_DRAWS,'bootstrap_seed':BOOTSTRAP_SEED})
boot=pd.DataFrame(boot);boot.to_csv(OUT/'bootstrap_summary.csv',index=False)
sep=pd.DataFrame(sep);sep.to_csv(OUT/'per_order_leaders.csv',index=False)
starting=[]
for group in range(4):
 rec={'first_task':SCENARIOS[group],'order_count':int((first==group).sum())}
 for m in METRICS:rec[m]=per_order.loc[first==group,[m+'__'+a for a in ALGOS]].values.mean()
 for m in ['FDE','MR']:
  for stat in ['initial_error','final_mean_error','BWT_post_error','BWT_reference_error','CT','CT_warm_start']:
   rec[m+'_'+stat]=extras[(extras.metric==m)&(extras.statistic==stat)&extras.Order.isin(np.where(first==group)[0])].value.mean()
 starting.append(rec)
pd.DataFrame(starting).to_csv(OUT/'starting_task_summary.csv',index=False)
# Distance-estimation sensitivity: legacy directed, legacy symmetric, and reconstructed primary.
sens=[]
for label,col in [('legacy_directed','legacy'),('legacy_symmetric','symmetrized'),('primary_raw_trajectory','corrected')]:
 Ms=matrix(CANDIDATES[0][1],col);ds=descriptor(Ms)
 for m in METRICS:
  for a in ALGOS:sens.append({'distance_version':label,'metric':m,'algorithm':a,'rho':rho(ds,per_order[m+'__'+a]),'unique_D_values':len(np.unique(ds))})
pd.DataFrame(sens).to_csv(OUT/'distance_sensitivity.csv',index=False)
# Paper tables. Three training replicates => sample SD; values remain in the logged units.
metric_tex={'FDE-CT':'FDE-CT','MR-CT':'MR-CT','FDE-BWT':'FDE-BWT','MR-BWT':'MR-BWT','FDE-CT2N':r'FDE-CT$_{2:N}$','MR-CT2N':r'MR-CT$_{2:N}$'}
def num(v,fmt='.2f',sign=False):
 """Typeset a number in math mode so that negative values carry a real minus sign."""
 s=format(v,('+' if sign else '')+fmt)
 return '$'+s+'$'
def write_table(name,lines): (OUT/name).write_text('\n'.join(lines)+'\n')
def boldmath(cell):
 """Bold a math-mode cell; \\textbf has no effect inside math."""
 return r'{\boldmath'+cell+'}'
sep_lookup={(r.metric,r.Order):r for r in sep.itertuples()}
for kind,ms in [('plasticity',['FDE-CT','MR-CT']),('stability',['FDE-BWT','MR-BWT'])]:
 cap=('Current-task errors by task order (mean $\\pm$ sample standard deviation over three replicates). FDE in metres; MR in percent. Bold: per-order minimum separable from the runner-up (Section~\\ref{subsec:analysis_protocol}).' if kind=='plasticity' else 'Backward transfer by task order (mean $\\pm$ sample standard deviation over three replicates). FDE in metres; MR in percentage points. Bold as in Table~\\ref{tab:main_results_plasticity}.')
 lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+cap+'}',r'\label{tab:main_results_'+kind+'}',r'\setlength{\tabcolsep}{3.5pt}',r'\renewcommand{\arraystretch}{1.0}',r'\fontsize{7}{7.7}\selectfont',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc}',r'\toprule',r'Order & \multicolumn{2}{c}{Vanilla} & \multicolumn{2}{c}{DER} & \multicolumn{2}{c}{SyReM} & \multicolumn{2}{c}{GSM} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', ' & '+' & '.join(metric_tex[m] for a in ALGOS for m in ms)+r' \\',r'\midrule']
 for oid in range(24):
  cellsx=[]
  for a in ALGOS:
   for m in ms:
    v=f'{means.loc[(oid,m,a)]:.2f}';s=f'{stds.loc[(oid,m,a)]:.2f}';lead=sep_lookup[(m,oid)]
    body='$'+v+r'_{\pm'+s+'}$'
    cellsx.append(boldmath(body) if (lead.separable and lead.leader==a) else body)
  lines.append(seq_letters[oid]+' & '+' & '.join(cellsx)+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}']);write_table('table_main_results_'+kind+'.tex',lines)
# Candidate descriptor correlations as a single-column table.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Spearman correlations between candidate transition-direction descriptors and order-level performance. W1: Wasserstein-1. Bold: $|\rho|\geq0.7$.}',r'\label{tab:correlation_comparison}',r'\setlength{\tabcolsep}{2.6pt}',r'\renewcommand{\arraystretch}{1.02}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrrr@{}}',r'\toprule',r'Algorithm & Metric & \shortstack{W1\\speed} & \shortstack{KL\\speed} & \shortstack{JS\\speed} & \shortstack{MMD\\speed} & \shortstack{W1\\accel.} \\',r'\midrule']
for a in ALGOS:
 for j,m in enumerate(METRICS):
  cellsx=[]
  for c,k in CANDIDATES:
   r=float(corr[(corr.candidate==c)&(corr.metric==m)&(corr.algorithm==a)].rho_all.iloc[0]);v=num(r,'.2f',True);cellsx.append(boldmath(v) if abs(r)>=.7 else v)
  lines.append((LABEL[a] if j==0 else '')+' & '+metric_tex[m]+' & '+' & '.join(cellsx)+r' \\')
 lines.append(r'\midrule')
for view in ['all']:
 vals=[next(x['mean_abs_rho_'+view] for x in summary if x['candidate']==c) for c,k in CANDIDATES]
 lines.append(r'\multicolumn{2}{l}{Mean $|\rho|$}'+' & '+' & '.join(num(v) for v in vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_correlation_comparison.tex',lines)
# Uniform and descriptor-weighted scores with bootstrap leader probabilities.
wdf=pd.DataFrame(weighted)
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Policy scores (units as in Tables~\ref{tab:main_results_plasticity} and~\ref{tab:main_results_stability}) under the uniform policy, the descriptor-weighted policy, and the uniform policy over the 18 orders that do not start with A, with the bootstrap probability of the smallest score under that policy in parentheses. Bold: smallest score per row.}',r'\label{tab:weighted_metrics}',r'\setlength{\tabcolsep}{1.5pt}',r'\scriptsize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrr@{}}',r'\toprule',r'Metric & Policy & Vanilla & DER & SyReM & GSM \\',r'\midrule']
for m in METRICS:
 rows=[]
 for scol,pcol,lab in [('naive_mean','p_best_uniform','Uniform'),('D_weighted','p_best_D_weighted','$D$-weighted'),('uniform_mean_excl_A_first','p_best_uniform_excl_A_first','Excl.\\ A-first')]:
  src=wdf if scol in wdf.columns else boot
  vals=[float(src[(src.metric==m)&(src.algorithm==a)][scol].iloc[0]) for a in ALGOS]
  probs=[float(boot[(boot.metric==m)&(boot.algorithm==a)][pcol].iloc[0]) for a in ALGOS]
  rows.append((lab,[(boldmath(num(v)) if v==min(vals) else num(v))+' ('+f'{pr:.2f}'+')' for v,pr in zip(vals,probs)]))
 for j,(lab,cellsx) in enumerate(rows):lines.append((metric_tex[m] if j==0 else '')+' & '+lab+' & '+' & '.join(cellsx)+r' \\')
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular*}',r'\end{table}']);write_table('table_weighted_metrics.tex',lines)
# Starting-task diagnostics: initial error, CT with and without the first stage, BWT components, final error.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{FDE (m) by starting task over 72 runs per row: initial error $R_{1,1}$, CT with and without (CT$_{2:N}$) the first stage, subsequent error $P$, reference $B$, BWT $=P-B$, and final error.}',r'\label{tab:bwt_diagnostic}',r'\setlength{\tabcolsep}{2.5pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}',r'\toprule',r'Start & $R_{1,1}$ & CT & CT$_{2:N}$ & $P$ & $B$ & BWT & Final \\',r'\midrule']
for i,rec in enumerate(starting):
 values=[rec['FDE_initial_error'],rec['FDE_CT'],rec['FDE_CT_warm_start'],rec['FDE_BWT_post_error'],rec['FDE_BWT_reference_error'],rec['FDE-BWT'],rec['FDE_final_mean_error']]
 lines.append('ABCD'[i]+' & '+' & '.join(num(v) for v in values)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_diagnostics.tex',lines)
# Final errors with bootstrap leader probabilities.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Final errors: mean [minimum, maximum] over order-level replicate means, and the bootstrap probability (Prob.) of the smallest uniform mean. FDE in metres; MR in percent. Bold: smallest mean.}',r'\label{tab:final_error}',r'\setlength{\tabcolsep}{3pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcrcr@{}}',r'\toprule',r'Algorithm & FDE: mean [min, max] & Prob. & MR: mean [min, max] & Prob. \\',r'\midrule']
exdf=pd.DataFrame(exsumm)
for a in ALGOS:
 vals=[]
 for m in ['FDE','MR']:
  rec=exdf[(exdf.algorithm==a)&(exdf.metric==m)&(exdf.statistic=='final_mean_error')].iloc[0]
  best=exdf[(exdf.metric==m)&(exdf.statistic=='final_mean_error')]['mean'].min()
  cell=f'{rec["mean"]:.2f} [{rec.order_min:.2f}, {rec.order_max:.2f}]'
  vals.append(r'\textbf{'+cell+'}' if rec['mean']==best else cell)
  vals.append(num(float(boot[(boot.metric==m+'-Final')&(boot.algorithm==a)].p_best_uniform.iloc[0])))
 lines.append(LABEL[a]+' & '+' & '.join(vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_final.tex',lines)
# Descriptor associations with and without adjustment for the starting task and for the position of A.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Mean Spearman associations across algorithms. $\rho(D)$: descriptor and order means. Excl.\ A-first: 18 orders. Start-adj.: rank residuals after removing starting-task means. $\rho(p_A)$: position of task A. $p_A$-adj.: rank residuals after removing position-of-A means.}',r'\label{tab:starting_correlations}',r'\setlength{\tabcolsep}{2.5pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',r'Metric & $\rho(D)$ & Excl.\ A-first & Start-adj. & $\rho(p_A)$ & $p_A$-adj. \\',r'\midrule']
pc=poscor.groupby('metric')[['rho_D','rho_D_without_A_first','rho_D_first_task_adjusted','rho_posA','rho_D_positionA_adjusted']].mean()
for m in METRICS+['FDE-CT2N','MR-CT2N']:
 if m=='FDE-CT2N':lines.append(r'\midrule')
 lines.append(metric_tex[m]+' & '+' & '.join(num(v,'.2f',True) for v in pc.loc[m].values)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_starting_correlations.tex',lines)
# Extended sum-of-squares decomposition table.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Percentage of the total centred sum of squares per component. The order term is split into starting-task identity (Start) and order within starting task; the last three columns give the 18 orders that do not start with A.}',r'\label{tab:decomposition}',r'\setlength{\tabcolsep}{1.8pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}',r'\toprule',r' & \multicolumn{5}{c}{All 24 orders (\%)} & \multicolumn{3}{c}{Excl.\ A-first (\%)} \\',r'\cmidrule(lr){2-6}\cmidrule(lr){7-9}',r'Metric & Start & \shortstack{Order$\mid$\\start} & Alg. & \shortstack{Order$\times$\\alg.} & Within & Order & Alg. & Within \\',r'\midrule']
for m in METRICS+['FDE-CT2N','MR-CT2N']:
 if m=='FDE-CT2N':lines.append(r'\midrule')
 ra=extended[(extended.metric==m)&(extended.subset=='all_orders')].iloc[0];rb=extended[(extended.metric==m)&(extended.subset=='excluding_A_first')].iloc[0]
 vals=[ra.start,ra.order_within_start,ra.algorithm,ra.order_x_algorithm,ra.within_cell_training,rb.order,rb.algorithm,rb.within_cell_training]
 lines.append(metric_tex[m]+' & '+' & '.join(num(100*v,'.1f') for v in vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_decomposition.tex',lines)
# Check whether a nominal replicate is a matched initial-state control across orders.
initial=raw[(raw.stage==1)&(raw.task==1)].merge(orders[['Order','first_task']],on='Order')
initial.to_csv(OUT/'initial_stage_across_orders.csv',index=False)
initial_blocks=initial.groupby(['first_task','seed','algorithm']).FDE.agg(['min','max','mean','std'])
initial_blocks['range']=initial_blocks['max']-initial_blocks['min'];initial_blocks.to_csv(OUT/'initial_stage_blocks.csv')
checks={'trajectories':int(perf.groupby(['seed','Order','algorithm']).ngroups),'stages_per_trajectory':4,'stage_task_rows':len(raw),'finite_values':bool(np.isfinite(raw[['FDE','MR']]).all().all()),'symmetric_distance':bool(np.array_equal(M,M.T)),'distinct_D_values':int(len(np.unique(D))),'D_values':sorted(np.unique(D).tolist()),'D_counts':pd.Series(D).value_counts().sort_index().tolist(),'D_formula':'(Step3-Step1)/2','sign_D_equals_A_in_positions_1_2':True,'spearman_D_positionA':rho(D,posA),'sample_SD_ddof':1,'initial_stage_nonidentical_blocks':int((initial_blocks['range']>1e-9).sum()),'initial_stage_total_blocks':len(initial_blocks),'first_task_adjustment':'Pearson correlation of residualized global average ranks after projecting each ranked variable onto four first-task indicator columns.','positionA_adjustment':'Same residualization with four position-of-A indicator columns.','bootstrap':{'draws':BOOTSTRAP_DRAWS,'seed':BOOTSTRAP_SEED,'resampling':'three replicates with replacement within every order-algorithm cell'},'separable_leader_rule':'per-order minimum whose gap to the runner-up exceeds the pooled sample SD of the two algorithms','primary_distance_units':'W1 speed in m/s; W1 acceleration in m/s^2; KL, JS and standardized-speed MMD dimensionless. See inputs/distance_parameters.json.','legacy_distance_sensitivity':'Preserved original directed and symmetric legacy matrices; primary full-array W1 ranks agree with symmetric legacy W1.'}
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
print('Recomputed',OUT)
print(pd.DataFrame(summary).to_string(index=False))
print(pc.to_string())
print(extended.to_string(index=False))
print(boot[['metric','algorithm','uniform_mean','p_best_uniform','p_best_D_weighted','winner_count','winner_count_q025','winner_count_q975','separable_leader_orders','separable_orders_total']].to_string(index=False))
print(pd.DataFrame(starting).to_string(index=False))
