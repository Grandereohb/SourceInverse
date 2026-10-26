"""Audit measured-case fits without inventing source or emission-rate truth."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pinn_source'))
from geo_utils import xy_to_latlon
METHODS=('ia_vpdi','matched_joint_adam')


def summarize_pairs(pairs,repeats=10000):
    """Whole-support-cluster resampling; not independent per-hour observations."""
    clusters=[g.normalized_rmse_IA_minus_joint.to_numpy() for _,g in
              pairs.groupby('temporal_cluster',sort=True)]
    if len(clusters)<2:
        return dict(cases=len(pairs),temporal_clusters=len(clusters),
            interval_low=None,interval_high=None,
            interval_method='Unavailable: fewer than two temporal-support clusters')
    rng=np.random.default_rng(20261019)
    estimates=[]
    for _ in range(repeats):
        selected=rng.integers(len(clusters),size=len(clusters))
        estimates.append(float(np.median(np.concatenate([clusters[i] for i in selected]))))
    low,high=np.quantile(estimates,[.025,.975])
    return dict(cases=len(pairs),temporal_clusters=len(clusters),
        normalized_rmse_difference_median=float(pairs.normalized_rmse_IA_minus_joint.median()),
        interval_low=float(low),interval_high=float(high),
        interval_method=f'95% percentile whole-support-cluster bootstrap, {repeats} repeats, seed20261019',
        scope='Conditional descriptive method contrast on this observational catalog, not source-location accuracy')


def analyze(study):
    study=Path(study).resolve()
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    preparation=ROOT/protocol['preparation'].replace('\\','/')
    prepared_path=preparation/'PREPARATION_FINISHED.json'
    if hashlib.sha256(prepared_path.read_bytes()).hexdigest()!=protocol['prepared_manifest_sha256']:
        raise ValueError('Prepared cohort manifest changed')
    prepared=json.loads(prepared_path.read_text(encoding='utf-8'))
    for name,value in prepared['prepared_sha256'].items():
        if hashlib.sha256((preparation/name.replace('\\','/')).read_bytes()).hexdigest()!=value:
            raise ValueError('Prepared input changed: '+name)
    rows,states,pairs,information=[],[],[],[]
    for case_id in protocol['case_ids']:
        case=study/case_id
        path=case/'result.json'
        stopped=(case/'STOPPED.json').exists()
        if not path.exists():
            states.append(dict(case_id=case_id,state='stopped' if stopped else 'pending',
                detail=json.loads((case/'STOPPED.json').read_text()) if stopped else None))
            continue
        record=json.loads(path.read_text(encoding='utf-8'))
        meta=json.loads((preparation/case_id/'input.json').read_text(encoding='utf-8'))
        if (record['case_id']!=case_id or record['metadata']!=meta or
                record['mode']!=protocol['mode'] or record['source_truth_available'] is not False
                or set(record['methods'])!=set(METHODS)):
            raise ValueError('Case identity, unknown-truth scope or method pairing mismatch: '+case_id)
        with np.load(preparation/case_id/'input.npz') as data:
            observed=data['normalized_concentration'][data['observation_mask']]
        scale=meta['concentration_scale']
        numerical=True
        local={}
        for method,value in record['methods'].items():
            source=np.asarray(value['source_norm'])
            coeff=np.asarray(value['coefficients'])
            prediction=np.asarray(value['prediction_normalized'])
            diffusivity=np.asarray(value['diffusion_m2_s'])
            if (source.shape!=(2,) or np.any(np.abs(source)>.750001) or coeff.shape!=(5,) or
                    prediction.shape!=observed.shape or np.any(coeff<0) or np.any(prediction<0) or
                    not all(np.isfinite(v).all() for v in (source,coeff,prediction,diffusivity))
                    or not diffusivity[0]>=diffusivity[1]>0):
                raise ValueError('Invalid numeric estimate: '+case_id+'/'+method)
            rmse=float(np.sqrt(np.mean((prediction-observed)**2)))
            ratio=float(prediction[np.argmax(observed)]/observed.max())
            np.testing.assert_allclose([rmse,rmse*scale,ratio],
                [value['normalized_rmse'],value['concentration_rmse_ug_m3'],value['peak_prediction_to_observation']],
                rtol=1e-10,atol=1e-10)
            normalization=record['normalization']
            np.testing.assert_allclose(np.asarray(value['diffusion_norm'])*normalization['length_m']**2/3600.,diffusivity,
                                       rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(xy_to_latlon(*(source*normalization['length_m']),
                                                   normalization['lon0'],normalization['lat0']),
                [value['longitude'],value['latitude']],rtol=0.,atol=1e-10)
            relative=value['numerical_replay_relative_difference']
            reference=value['reference_release_relative_difference']
            if (not np.isfinite([relative,reference]).all() or min(relative,reference)<0 or
                    value['numerical_gate_passed']!=(relative<=.01 and reference<=.0025)):
                raise ValueError('Numerical replay status inconsistent with measured differences')
            mean_q=float(np.trapezoid(coeff,np.linspace(0.,meta['n_hours'],5))/meta['n_hours'])
            row=dict(case_id=case_id,temporal_cluster=meta['temporal_cluster'],pollutant=meta['pollutant'],
                method=method,n_stations=meta['n_stations'],n_hours=meta['n_hours'],
                n_observations=len(observed),normalized_rmse=rmse,concentration_rmse_ug_m3=rmse*scale,
                peak_prediction_to_observation=ratio,absolute_peak_ratio_error=abs(ratio-1),
                source_longitude=value['longitude'],source_latitude=value['latitude'],
                near_source_search_boundary=value['near_source_search_boundary'],
                effective_q_mean_model_units=mean_q,effective_q_max_model_units=float(coeff.max()),
                seconds=value['seconds_including_common_search'],convergence=value['convergence'],
                numerical_replay_relative_difference=value['numerical_replay_relative_difference'],
                reference_release_relative_difference=value['reference_release_relative_difference'],
                numerical_gate_passed=value['numerical_gate_passed'])
            if not np.isfinite(row['seconds']) or row['seconds']<0:
                raise ValueError('Invalid recorded runtime')
            local[method]=row
            rows.append(row)
            numerical=numerical and row['numerical_gate_passed']
        ia,joint=local['ia_vpdi'],local['matched_joint_adam']
        source_ia=np.asarray(record['methods']['ia_vpdi']['source_norm'])
        source_joint=np.asarray(record['methods']['matched_joint_adam']['source_norm'])
        pairs.append(dict(case_id=case_id,temporal_cluster=meta['temporal_cluster'],pollutant=meta['pollutant'],
            normalized_rmse_IA_minus_joint=ia['normalized_rmse']-joint['normalized_rmse'],
            absolute_peak_error_IA_minus_joint=ia['absolute_peak_ratio_error']-joint['absolute_peak_ratio_error'],
            method_position_disagreement_m=float(np.linalg.norm(source_ia-source_joint)*record['normalization']['length_m'])))
        info=record['conditional_information']
        information.append(dict(case_id=case_id,temporal_cluster=meta['temporal_cluster'],
            information_status=info['status'],rank=info['rank'],condition_number=info['condition_number'],
            ellipse_major_m=info['ellipse_major_m'],ellipse_minor_m=info['ellipse_minor_m'],
            boundary_sensitive=info['boundary_sensitive'],
            interpretation='Conditional local model diagnostic, NOT verified source accuracy or calibrated confidence'))
        states.append(dict(case_id=case_id,state='stopped' if stopped else 'completed' if numerical else 'numerical_failure'))
    output=study/'analysis'
    output.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'case_metrics.csv',index=False)
    pd.DataFrame(states).to_csv(output/'case_status.csv',index=False)
    paired=pd.DataFrame(pairs)
    paired.to_csv(output/'paired_methods.csv',index=False)
    pd.DataFrame(information).to_csv(output/'local_diagnostics.csv',index=False)
    completed=sum(v['state']=='completed' for v in states)
    final=protocol['mode']=='full' and completed==len(protocol['case_ids'])
    status=dict(mode=protocol['mode'],final=final,planned_cases=len(protocol['case_ids']),completed_cases=completed,
        stopped_cases=[v['case_id'] for v in states if v['state']=='stopped'],
        numerical_failure_cases=[v['case_id'] for v in states if v['state']=='numerical_failure'],
        source_truth_available=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cautions=['No industrial location/rate error can be evaluated without truth',
                  'Instrument flags and coordinate datum remain uncertified',
                  'Local diagnostic labels do not certify source attribution',
                  'No manuscript aggregate or complete-cohort figure from partial or smoke runs',
                  '59 tasks share21 temporal-support clusters and one station system',
                  'Runtimes follow fixed IA-then-joint order with common cache; not isolated repeated benchmarks'])
    (output/'analysis_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if final:
        frame.groupby('method',as_index=False).agg(n=('case_id','size'),
            normalized_rmse_median=('normalized_rmse','median'),
            normalized_rmse_q25=('normalized_rmse',lambda v:float(v.quantile(.25))),
            normalized_rmse_q75=('normalized_rmse',lambda v:float(v.quantile(.75))),
            absolute_peak_ratio_error_median=('absolute_peak_ratio_error','median'),
            boundary_count=('near_source_search_boundary','sum'),
            seconds_median=('seconds','median')).to_csv(output/'summary.csv',index=False)
        (output/'paired_cluster_effect.json').write_text(json.dumps(summarize_pairs(paired),indent=2),encoding='utf-8')
    return frame,status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args()
    frame,status=analyze(args.study)
    print(json.dumps(status,indent=2))
