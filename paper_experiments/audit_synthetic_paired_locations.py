"""Paired accuracy audit of the unchanged192-case nominal synthetic cohort."""

import hashlib

import numpy as np
import pandas as pd

from run_metec_comparison import ROOT,read,save

SOURCE=ROOT/'result/paper_experiments_20261017/paper_focused_20261027/analysis/case_flags.csv'
DEST=SOURCE.parents[2]/'synthetic_paired_accuracy_audit_20261028'


def run():
    full=pd.read_csv(SOURCE)
    frame=full[full.cohort.eq('nominal')].copy()
    paired=frame.pivot(index='case_id',columns='method',values='localization_error_m')
    metadata=frame[frame.method.eq('ia_vpdi')].set_index('case_id')
    if len(paired)!=192 or paired.isna().any().any() or not paired.index.is_unique:
        raise ValueError('Expected unchanged complete192-case paired cohort')
    difference=paired.ia_vpdi-paired.joint_nls_trf
    rng=np.random.default_rng(730001)
    repetitions=10000
    draws=np.zeros(repetitions)
    for _,group in metadata.groupby(['network','wind','release'],sort=True):
        values=difference.loc[group.index].to_numpy()
        draws+=rng.choice(values,size=(repetitions,len(values)),replace=True).sum(1)/len(paired)
    summary=frame.groupby('method').localization_error_m.agg(['count','median','mean'])
    DEST.mkdir(parents=True,exist_ok=True)
    summary.to_csv(DEST/'method_summary.csv',encoding='utf-8-sig')
    frame.groupby(['network','method']).localization_error_m.agg(['median','mean']).to_csv(
        DEST/'network_summary.csv',encoding='utf-8-sig')
    save(DEST/'summary.json',dict(cases=192,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        ia_mean_m=float(paired.ia_vpdi.mean()),nls_mean_m=float(paired.joint_nls_trf.mean()),
        paired_mean_difference_ia_minus_nls_m=float(difference.mean()),
        stratified_pair_bootstrap_95pct_m=np.quantile(draws,[.025,.975]),
        ia_better_by_more_than1m=int((difference< -1).sum()),
        nls_better_by_more_than1m=int((difference>1).sum()),
        within1m=int((difference.abs()<=1).sum()),
        ia_median_m=float(paired.ia_vpdi.median()),nls_median_m=float(paired.joint_nls_trf.median()),
        bootstrap=dict(seed=730001,repetitions=repetitions,strata=['network','wind','release'],
            unit='Paired synthetic case; resample within fixed scenario strata, not model-run repeats'),
        scope='Exploratory development-cohort summary conditional on these four layouts and this protocol',
        limitations=['Existing cases informed method development; not confirmatory independent validation',
            'Mean benefit can be driven by fewer severe failures; median precision is almost unchanged',
            'IA1/3/5 model selection vs fixed5 NLS; cannot attribute to elimination alone',
            'No claim of significant public-data improvement or generalization to new geometries'],
        refits_performed=False,full_goal_achieved=False))
    print(summary.to_string(),flush=True)
    print('PAIRED_MEAN_DIFFERENCE',float(difference.mean()),'CI',np.quantile(draws,[.025,.975]),flush=True)


if __name__=='__main__':
    run()
