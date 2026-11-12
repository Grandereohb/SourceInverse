"""Complete-cohort paired accuracy audit, separating wind-model and solver comparisons."""

import hashlib

import numpy as np
import pandas as pd

from run_ia_affine_wind_full import DEST
from run_metec_comparison import OUT,read,save


def intervals(difference,groups,rng,repetitions=10000):
    values=np.asarray(difference)
    event_draws=rng.choice(values,size=(repetitions,len(values)),replace=True).mean(1)
    clusters=[values[np.asarray(groups)==group] for group in np.unique(groups)]
    draws=rng.integers(0,len(clusters),size=(repetitions,len(clusters)))
    sums=np.array([cluster.sum() for cluster in clusters])
    sizes=np.array([len(cluster) for cluster in clusters])
    cluster_draws=sums[draws].sum(1)/sizes[draws].sum(1)
    return dict(mean_difference_m=float(values.mean()),median_difference_m=float(np.median(values)),
        event_bootstrap_95pct_m=np.quantile(event_draws,[.025,.975]),
        equipment_cluster_bootstrap_95pct_m=np.quantile(cluster_draws,[.025,.975]),
        ia_better_by_more_than1m=int((values< -1).sum()),
        comparator_better_by_more_than1m=int((values>1).sum()),within1m=int((np.abs(values)<=1).sum()))


def run():
    protocol=read(DEST/'protocol.json')
    status=read(DEST/'status.json')
    if status['complete_events']!=17 or status['failed_events']:
        raise ValueError('Do not summarize a favourable incomplete subset')
    frame=pd.read_csv(DEST/'comparison.csv',dtype={'event_id':str})
    wide=frame.pivot(index='event_id',columns='method',values='localization_error_m')
    if len(wide)!=17 or wide.isna().any().any():
        raise ValueError('Every method must cover every frozen public event')
    groups=[read(OUT/'inputs'/identifier/'truth.json')['equipment_group'] for identifier in wide.index]
    rng=np.random.default_rng(20261028)
    result=dict(events=17,equipment_groups=len(set(groups)),seed=20261028,repetitions=10000,
        physical_model_comparison=intervals(wide.ia_affine_wind-wide.joint_nls_trf,groups,rng),
        matched_wind_comparison=intervals(wide.ia_affine_wind-wide.joint_nls_affine_wind,groups,rng),
        original_ia_comparison=intervals(wide.ia_affine_wind-wide.ia_vpdi,groups,rng),
        interpretation='Negative differences favour IA; physical-model and same-model comparisons are distinct',
        inference_scope='Exploratory fixed-site development data;5 equipment groups,one week; not independent-site confirmation',
        numerical_ready=bool(status.get('ready',False)),
        numerical_replay_pending=bool(status.get('replay_pending',True)),full_goal_achieved=False,
        limitations=['Small cluster count; intervals are exploratory, not generalization guarantees',
            'Event bootstrap ignores persistent equipment/weather dependence; also report cluster bootstrap',
            'Capacity selection1/3/5 versus fixed5 is not a pure variable-elimination ablation'],
        source_comparison_sha256=hashlib.sha256((DEST/'comparison.csv').read_bytes()).hexdigest())
    save(DEST/'paired_accuracy.json',result)
    paradox=[]
    for identifier in wide.index:
        ia=read(DEST/identifier/'ia_affine_wind.json')
        nls=read(DEST/identifier/'joint_nls_affine_wind.json')
        paradox.append(dict(event_id=identifier,q_basis_count=ia['q_basis_count'],
            ia_sse=ia['sse'],nls_sse=nls['sse'],
            ia_location_m=wide.loc[identifier,'ia_affine_wind'],
            nls_location_m=wide.loc[identifier,'joint_nls_affine_wind'],
            lower_ia_sse_but_worse_location=bool(ia['sse']<nls['sse'] and
                wide.loc[identifier,'ia_affine_wind']>wide.loc[identifier,'joint_nls_affine_wind']+1.)))
    pd.DataFrame(paradox).to_csv(DEST/'fit_location_paradox.csv',index=False,encoding='utf-8-sig')
    print('MATCHED_WIND',result['matched_wind_comparison'],flush=True)
    print('ORIGINAL_NLS',result['physical_model_comparison'],flush=True)
    print('NUMERICAL_READY',result['numerical_ready'],flush=True)


if __name__=='__main__':
    run()
