#!/usr/bin/env python3
"""Recompute the descriptive paper analysis from frozen CSV inputs.

No model training, pickle, stochastic resampling, or inferential p-values.
Primary distances are reconstructed from numeric trajectories using the frozen
distance_parameters.json. KL remains directional; the other matrices are symmetric. D=(last transition-first transition)/2 is evaluated
algebraically, retaining exact ties. Scenario order is A,B,C,D as below.
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
orders=pd.read_csv(IN/'orders.csv').sort_values('Order')
seqs=[[SCENARIOS.index(s) for s in v.split('|')] for v in orders.scenario_sequence]
first=np.array([s[0] for s in seqs]);no_A=first!=0
perf=pd.read_csv(IN/'performance_per_seed.csv')
raw=pd.read_csv(IN/'stage_task_results.csv')
dist=pd.read_csv(IN/'distance_matrices.csv')
def matrix(key,col='corrected'):
 return dist[dist.distance==key].pivot(index='from',columns='to',values=col).reindex(index=SCENARIOS,columns=SCENARIOS).values
def descriptor(M):return np.array([(M[o[2],o[3]]-M[o[0],o[1]])/2 for o in seqs])
def rho(x,y):return float(spearmanr(x,y).statistic)
def group_residual(x,rank=False):
 v=rankdata(x) if rank else np.asarray(x,dtype=float)
 return v-np.array([v[first==g].mean() for g in first])
def adjusted_rho(x,y):return float(pearsonr(group_residual(x,True),group_residual(y,True)).statistic)
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
M=matrix(CANDIDATES[0][1]);D=descriptor(M)
assert np.array_equal(M,M.T)
assert len(np.unique(D))==6
assert np.allclose(np.sort(D),-np.sort(D)[::-1])
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
variance=[];finite=[];algmeans=[]
for m in METRICS:
 x=np.empty((24,4,3))
 for ai,a in enumerate(ALGOS):
  x[:,ai,:]=perf[(perf.metric==m)&(perf.algorithm==a)].pivot(index='Order',columns='seed',values='value').reindex(range(24)).values
 grand=x.mean();SSt,pieces=decompose_variation(x)
 for term,ss in pieces.items():variance.append({'metric':m,'component':term,'SS':ss,'fraction_total_SS':ss/SSt,'total_SS':SSt})
 y=x.mean(2);wins=np.argmin(y,axis=1)
 finite.append({'metric':m,'grand_mean':grand,'per_order_algorithm_mean_min':y.mean(1).min(),'per_order_algorithm_mean_max':y.mean(1).max(),'per_order_algorithm_mean_range':np.ptp(y.mean(1)),'median_within_order_algorithm_range':np.median(np.ptp(y,axis=1)),'algorithm_mean_gap':np.ptp(y.mean(0))})
 for ai,a in enumerate(ALGOS):algmeans.append({'metric':m,'algorithm':a,'mean':y[:,ai].mean(),'order_min':y[:,ai].min(),'order_max':y[:,ai].max(),'winner_count':int(np.sum(wins==ai))})
pd.DataFrame(variance).to_csv(OUT/'sum_of_squares.csv',index=False)
pd.DataFrame(finite).to_csv(OUT/'finite_order_summary.csv',index=False)
pd.DataFrame(algmeans).to_csv(OUT/'algorithm_summary.csv',index=False)
# Raw-log diagnostics: initial task error, final mean error, BWT numerator components.
extra=[]
for (seed,oid,a),g in raw.groupby(['seed','Order','algorithm']):
 for m in ['FDE','MR']:
  R=g.pivot(index='stage',columns='task',values=m).values
  post=np.mean([R[r,:r].mean() for r in range(1,4)])
  reference=np.mean([np.diag(R)[:r].mean() for r in range(1,4)])
  values={'initial_error':R[0,0],'final_mean_error':R[-1].mean(),'BWT_post_error':post,'BWT_reference_error':reference,'BWT':post-reference,'CT':np.diag(R).mean()}
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
starting=[]
for group in range(4):
 rec={'first_task':SCENARIOS[group],'order_count':int((first==group).sum())}
 for m in METRICS:rec[m]=per_order.loc[first==group,[m+'__'+a for a in ALGOS]].values.mean()
 for m in ['FDE','MR']:
  for stat in ['initial_error','final_mean_error','BWT_post_error','BWT_reference_error']:
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
metric_tex={'FDE-CT':r'FDE$_{\rm CT}$','MR-CT':r'MR$_{\rm CT}$','FDE-BWT':r'FDE$_{\rm BWT}$','MR-BWT':r'MR$_{\rm BWT}$'}
def write_table(name,lines): (OUT/name).write_text('\n'.join(lines)+'\n')
for kind,ms in [('plasticity',['FDE-CT','MR-CT']),('stability',['FDE-BWT','MR-BWT'])]:
 cap=('Current-task errors by task order (mean $\\pm$ sample SD). FDE in metres; MR in percent.' if kind=='plasticity' else 'Backward transfer by task order (mean $\\pm$ sample SD). FDE in metres; MR in percentage points.')
 lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+cap+'}',r'\label{tab:main_results_'+kind+'}',r'\setlength{\tabcolsep}{4.0pt}',r'\renewcommand{\arraystretch}{1.0}',r'\fontsize{7.5}{8.5}\selectfont',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc}',r'\toprule',r'Order & \multicolumn{2}{c}{Vanilla} & \multicolumn{2}{c}{DER} & \multicolumn{2}{c}{SyReM} & \multicolumn{2}{c}{GSM} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', ' & '+' & '.join(metric_tex[m] for a in ALGOS for m in ms)+r' \\',r'\midrule']
 for oid in range(24):lines.append(str(oid+1)+' & '+' & '.join(f'{means.loc[(oid,m,a)]:.2f}'+r'$_{\pm'+f'{stds.loc[(oid,m,a)]:.2f}'+'}$' for a in ALGOS for m in ms)+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}']);write_table('table_main_results_'+kind+'.tex',lines)
lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{Spearman correlations for five candidate transition descriptors. Bold: $|\rho|\geq0.7$.}',r'\label{tab:correlation_comparison}',r'\setlength{\tabcolsep}{7pt}',r'\renewcommand{\arraystretch}{1.04}',r'\footnotesize',r'\begin{tabular}{llccccc}',r'\toprule',r'Algorithm & Metric & W1 speed & KL speed & JS speed & MMD speed & W1 accel. \\',r'\midrule']
for a in ALGOS:
 for j,m in enumerate(METRICS):
  cells=[]
  for c,k in CANDIDATES:
   r=float(corr[(corr.candidate==c)&(corr.metric==m)&(corr.algorithm==a)].rho_all.iloc[0]);v=f'{r:+.2f}';cells.append(r'\textbf{'+v+'}' if abs(r)>=.7 else v)
  lines.append((LABEL[a] if j==0 else '')+' & '+metric_tex[m]+' & '+' & '.join(cells)+r' \\')
 lines.append(r'\midrule')
for view in ['CT','BWT','all']:
 vals=[next(x['mean_abs_rho_'+view] for x in summary if x['candidate']==c) for c,k in CANDIDATES]
 lines.append(r'\multicolumn{2}{l}{Mean $|\rho|$, '+view+'}'+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write_table('table_correlation_comparison.tex',lines)
wdf=pd.DataFrame(weighted)
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Uniform and contrast-weighted scores. Bold marks the smallest value in each row.}',r'\label{tab:weighted_metrics}',r'\setlength{\tabcolsep}{3.5pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrr@{}}',r'\toprule',r'Metric & Weight & Vanilla & DER & SyReM & GSM \\',r'\midrule']
for m in METRICS:
 for j,(col,lab) in enumerate([('naive_mean','Uniform'),('D_weighted','$D$-weighted')]):
  vals=[float(wdf[(wdf.metric==m)&(wdf.algorithm==a)][col].iloc[0]) for a in ALGOS]
  cells=[r'\textbf{'+f'{v:.2f}'+'}' if v==min(vals) else f'{v:.2f}' for v in vals]
  lines.append((metric_tex[m] if j==0 else '')+' & '+lab+' & '+' & '.join(cells)+r' \\')
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular*}',r'\end{table}']);write_table('table_weighted_metrics.tex',lines)
# Compact supporting tables for the first-task and final-error diagnostics.
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{FDE decomposition by starting task (metres), averaged over 72 training runs per row.}',r'\label{tab:bwt_diagnostic}',r'\setlength{\tabcolsep}{3.5pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',r'Start & $R_{1,1}$ & $P$ & $B$ & BWT & Final mean \\',r'\midrule']
for i,rec in enumerate(starting):
 values=[rec['FDE_initial_error'],rec['FDE_BWT_post_error'],rec['FDE_BWT_reference_error'],rec['FDE-BWT'],rec['FDE_final_mean_error']]
 lines.append('ABCD'[i]+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_diagnostics.tex',lines)
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Final errors: mean [minimum, maximum] over order-level replicate means. FDE in metres; MR in percent.}',r'\label{tab:final_error}',r'\setlength{\tabcolsep}{4pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}',r'\toprule',r'Algorithm & FDE: mean [min, max] & MR: mean [min, max] \\',r'\midrule']
exdf=pd.DataFrame(exsumm)
for a in ALGOS:
 vals=[]
 for m in ['FDE','MR']:
  rec=exdf[(exdf.algorithm==a)&(exdf.metric==m)&(exdf.statistic=='final_mean_error')].iloc[0]
  vals.append(f'{rec["mean"]:.2f} [{rec.order_min:.2f}, {rec.order_max:.2f}]')
 lines.append(LABEL[a]+' & '+' & '.join(vals)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_final.tex',lines)
lines=[r'\begin{table}[!t]',r'\centering',r'\caption{Mean descriptor correlations across algorithms, with and without adjustment for the starting task.}',r'\label{tab:starting_correlations}',r'\setlength{\tabcolsep}{5pt}',r'\footnotesize',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}',r'\toprule',r'Metric & All orders & Excluding A-first & Adjusted \\',r'\midrule']
cs=corr[corr.candidate=='W1 speed'].groupby('metric')[['rho_all','rho_without_A','rho_first_task_adjusted']].mean()
for m in METRICS:lines.append(metric_tex[m]+' & '+' & '.join(f'{v:+.2f}' for v in cs.loc[m].values)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table}']);write_table('table_starting_correlations.tex',lines)
# Check whether a nominal replicate is a matched initial-state control across orders.
initial=raw[(raw.stage==1)&(raw.task==1)].merge(orders[['Order','first_task']],on='Order')
initial.to_csv(OUT/'initial_stage_across_orders.csv',index=False)
initial_blocks=initial.groupby(['first_task','seed','algorithm']).FDE.agg(['min','max','mean','std'])
initial_blocks['range']=initial_blocks['max']-initial_blocks['min'];initial_blocks.to_csv(OUT/'initial_stage_blocks.csv')
checks={'trajectories':int(perf.groupby(['seed','Order','algorithm']).ngroups),'stages_per_trajectory':4,'stage_task_rows':len(raw),'finite_values':bool(np.isfinite(raw[['FDE','MR']]).all().all()),'symmetric_distance':bool(np.array_equal(M,M.T)),'distinct_D_values':int(len(np.unique(D))),'D_values':sorted(np.unique(D).tolist()),'D_counts':pd.Series(D).value_counts().sort_index().tolist(),'D_formula':'(Step3-Step1)/2','sample_SD_ddof':1,'initial_stage_nonidentical_blocks':int((initial_blocks['range']>1e-9).sum()),'initial_stage_total_blocks':len(initial_blocks),'first_task_adjustment':'Pearson correlation of residualized global average ranks after projecting each ranked variable onto four first-task indicator columns.','primary_distance_units':'W1 speed in m/s; W1 acceleration in m/s^2; KL, JS and standardized-speed MMD dimensionless. See inputs/distance_parameters.json.','legacy_distance_sensitivity':'Preserved original directed and symmetric legacy matrices; primary full-array W1 ranks agree with symmetric legacy W1.'}
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
print('Recomputed',OUT)
print(pd.DataFrame(summary).to_string(index=False))
print(corr[corr.candidate=='W1 speed'].groupby('metric')[['rho_all','rho_without_A','rho_first_task_adjusted']].mean().to_string())
print(pd.DataFrame(starting).to_string(index=False))
