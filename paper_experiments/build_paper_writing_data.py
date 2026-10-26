"""Reproducible writing tables, retaining full cohorts and metric definitions."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'
OUTPUT=ROOT/'manuscript_results_v1'
DOI='https://doi.org/10.1088/1361-6420/adfb29'


def literature_tables(output=None):
    output=Path(output) if output is not None else OUTPUT
    # Experimental Case 2, Table 2; simulation rows are deliberately excluded.
    published=[
        dict(event='source2_rel06',release='rel06',reference_q_kg_h=1.35,
             d2_error=2.6,d3_error=1.1,d2_q=1.44,d3_q=1.17,d2_pct=7.,d3_pct=-13.,
             d2_plusminus=.07,d3_lower=.93,d3_upper=1.47),
        dict(event='source1_rel07',release='rel07',reference_q_kg_h=1.50,
             d2_error=12.1,d3_error=8.3,d2_q=1.33,d3_q=1.70,d2_pct=-12.,d3_pct=13.,
             d2_plusminus=.02,d3_lower=1.37,d3_upper=2.08)]
    own=pd.read_csv(ROOT/'controlled_release_fit_v2_precision/analysis/case_metrics.csv')
    rows=[]
    for source in published:
        manifest=json.loads((ROOT/'controlled_release_preparation_v2'/source['event']/'manifest.json').read_text(encoding='utf-8'))
        for dimension in ('2','3'):
            rows.append(dict(event=source['event'],release=source['release'],
                method=f'published_{dimension}D_BSE',height_m=np.nan,
                localization_error_m=source[f'd{dimension}_error'],
                estimated_q_kg_h=source[f'd{dimension}_q'],
                published_reference_q_kg_h=source['reference_q_kg_h'],
                published_signed_q_error_pct=source[f'd{dimension}_pct'],
                published_q_plusminus_kg_h=source['d2_plusminus'] if dimension=='2' else np.nan,
                published_q_95_lower_kg_h=source['d3_lower'] if dimension=='3' else np.nan,
                published_q_95_upper_kg_h=source['d3_upper'] if dimension=='3' else np.nan,
                published_q_uncertainty_type='reported plus-minus; not recast as a 95% interval' if dimension=='2' else '95% credible interval',
                own_signed_q_error_vs_published_reference_pct=np.nan,
                own_window_start='',own_window_end='',doi=DOI,
                source='Vanska et al. 2025, Inverse Problems, experimental Case2 Table2',
                comparability='Same release ID and point-location distance; not identical temporal processing or source priors'))
        subset=own[(own.event==source['event'])&(own.method=='transient_constant_q')]
        if len(subset)!=2:
            raise ValueError('Both own source heights must be retained')
        for _,row in subset.iterrows():
            rows.append(dict(event=source['event'],release=source['release'],method='IA_transient_constant_Q',
                height_m=row.height_m,localization_error_m=row.localization_error_m,
                estimated_q_kg_h=row.mean_q_kg_h,
                published_reference_q_kg_h=source['reference_q_kg_h'],
                published_signed_q_error_pct=np.nan,
                own_signed_q_error_vs_published_reference_pct=100*(row.mean_q_kg_h/source['reference_q_kg_h']-1),
                own_window_start=manifest['window_original_clock'][0],
                own_window_end=manifest['window_original_clock'][1],doi=DOI,
                source='Own saved fits; external reference is secondary published evidence, NOT independent weighing',
                comparability='Same release ID; own 1min averages, known single source, constant-Q physical adapter; paper3-5s state updates and distributed source mapping'))
    pd.DataFrame(rows).to_csv(output/'controlled_release_same_event_comparison.csv',index=False)
    overview=[
        dict(study='Hirst et al. 2020',method='Gaussian plume + MCMC',
             metric='reported source localization',value='single source: few meters; double-source offsets:6/12m',
             conditions='17 releases grouped into4 configurations, including multiple sources; not event-matched',
             doi='https://doi.org/10.1029/2019GL086725',evidence='Section4 and figures3/4',comparison_level='same dataset; different aggregation'),
        dict(study='Voss et al. 2024',method='2D Bayesian state estimation',
             metric='reported localization range; source-rate relative-error threshold',
             value='3-12m;80% of cases have source-rate error<30%',
             conditions='Temporal source-map reconstruction; retain paper statistics, not median ranking',
             doi='https://doi.org/10.1016/j.aeaoa.2024.100260',evidence='publisher abstract',comparison_level='same dataset; metric definition only partly aligned'),
        dict(study='Vanska et al. 2025',method='2D and3D Bayesian state estimation',
             metric='point localization distance per release; mean emission-rate estimate',
             value='rel06:2.6/1.1m;rel07:12.1/8.3m (2D/3D)',
             conditions='Case2 Table2; source-map temporal estimates;3-5s experimental steps; source constrained to ground layer',
             doi=DOI,evidence='experimental Case2,Table2',comparison_level='same release IDs and units; full processing not matched'),
        dict(study='Newman et al. 2025 AOAS',method='Gaussian plume + M-MALA within Gibbs',
             metric='concentration RMSE',value='Source1 estimated Smith:0.611;Source2 estimated Draxler:0.477',
             conditions='Aggregated source-specific releases; concentration residual, NOT location or source-rate error',
             doi='https://doi.org/10.1214/25-AOAS2101',evidence='Tables1/2; author main PDF pp15-16',comparison_level='same dataset, different metric and aggregation'),
        dict(study='Newman et al. 2025 preprint',method='CFD surrogate + SIR',
             metric='mean distance of final particles from true source',value='MLP:5.82m;Gaussian:11.09m',
             conditions='Source1 rel02 ten-minute subset; posterior-distribution metric, NOT point-estimator error',
             doi='https://arxiv.org/abs/2506.14597',evidence='Section4.2 Table2',comparison_level='reference only; not metric-matched')]
    pd.DataFrame(overview).to_csv(output/'literature_metrics_and_conditions.csv',index=False)


def build(supplement=None, output=None):
    output=Path(output) if output is not None else OUTPUT
    supplement=Path(supplement) if supplement is not None else ROOT/'supplementary_scenarios_v1'
    protocol=json.loads((supplement/'protocol.json').read_text(encoding='utf-8'))
    planned=int(protocol['case_count'])
    results=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(supplement.glob('supp_*/result.json'))]
    complete=[r for r in results if r.get('comparison_version')=='shared_forward_v3']
    if len(complete)!=planned:
        raise ValueError(f'Complete all {planned} supplementary cases before writing final tables')
    output.mkdir(exist_ok=True)
    main=ROOT/'evaluation_v3_shared_forward/analysis'
    input_hashes=json.loads((main.parent/'input_hash_manifest.json').read_text(encoding='utf-8'))['sha256']
    if len(input_hashes)!=240 or any(
            hashlib.sha256((main.parent/name).read_bytes()).hexdigest()!=expected
            for name,expected in input_hashes.items()):
        raise ValueError('Original 120-case input hashes changed')
    state=json.loads((main/'analysis_status.json').read_text(encoding='utf-8'))
    if not state['final'] or state['completed_cases']!=120:
        raise ValueError('Complete main cohort required')
    for name in ('case_metrics.csv','group_summary.csv','paired_effects.csv','local_diagnostics.csv'):
        shutil.copy2(main/name,output/('main_'+name))
    frame=pd.read_csv(main/'case_metrics.csv')
    if (len(frame)!=360 or frame.case_id.nunique()!=120 or
            frame.duplicated(['case_id','method']).any() or
            not np.isfinite(frame[['localization_error_m','q_relative_l2']].to_numpy()).all()):
        raise ValueError('Expected finite, unique results for all 120 cases and three methods')
    wide=frame.pivot(index='case_id',columns='method',values='localization_error_m')
    winners=wide.idxmin(axis=1).value_counts().to_dict()
    calibration=ROOT/'uncertainty_calibration_v1'
    for name in ('case_ellipses.csv','coverage_summary.csv','protocol.json','calibration_result.json'):
        destination=name if name=='calibration_result.json' else 'calibration_'+name
        shutil.copy2(calibration/name,output/destination)
    physical=ROOT/'controlled_release_fit_v2_precision/analysis'
    measured=pd.read_csv(physical/'case_metrics.csv')
    if (len(measured)!=48 or measured.event.nunique()!=8 or
            measured.duplicated(['event','height_m','method']).any() or
            not np.isfinite(measured[['localization_error_m','mean_q_kg_h']].to_numpy()).all()):
        raise ValueError('Expected all eight events, two heights and three finite model results')
    shutil.copy2(physical/'case_metrics.csv',output/'controlled_release_case_metrics.csv')
    shutil.copy2(physical/'summary.csv',output/'controlled_release_summary.csv')
    literature_tables(output)
    supplement_state=json.loads((supplement/'analysis/analysis_status.json').read_text(encoding='utf-8'))
    if not supplement_state['final'] or supplement_state['completed_cases']!=planned:
        raise ValueError('Supplement analysis is not final')
    for name in ('case_metrics.csv','group_summary.csv'):
        path=supplement/'analysis'/name
        shutil.copy2(path,output/('supplement_'+name))
    extra=pd.read_csv(output/'supplement_case_metrics.csv')
    if (len(extra)!=3*planned or extra.case_id.nunique()!=planned or
            extra.duplicated(['case_id','method']).any() or
            set(extra.case_id)!={s['case_id'] for s in protocol['case_design']} or
            not np.isfinite(extra[['localization_error_m','q_relative_l2']].to_numpy()).all()):
        raise ValueError('Expected every planned supplementary case and three finite methods')
    combined=pd.concat([frame.assign(source_study=main.parent.name),
                        extra.assign(source_study=supplement.name)],ignore_index=True)
    combined.to_csv(output/'combined_case_metrics.csv',index=False)
    references=[]
    for directory,pattern in [('evaluation_v3_shared_forward','eval_*'),
                              (supplement.name,'supp_*')]:
        protocol=ROOT/directory/'protocol.json'
        references.append(dict(path=protocol.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),bytes=protocol.stat().st_size))
        for case in sorted((ROOT/directory).glob(pattern)):
            for name in ('input.json','input.npz','result.json','optimization_traces.npz'):
                path=case/name
                if path.exists():
                    references.append(dict(path=path.relative_to(ROOT).as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    for directory in ('controlled_release_preparation_v2','controlled_release_fit_v2_precision',
                      'uncertainty_calibration_v1'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and 'source_snapshot' not in path.parts:
                references.append(dict(path=path.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    report=dict(main_completed=120,main_method_records=len(frame),main_winner_counts=winners,
        supplement_planned=planned,supplement_completed=len(complete),supplement_study=supplement.name,
        calibration_attempts=1,all_original_cases_preserved=True,
        original_input_hashes_verified=len(input_hashes),
        outcome_based_selection_used_for_ranking=False,
        supplemental_selection='Two input-defined scenarios; every planned seed retained',
        literature_matching='rel06/rel07 same-ID metric-aligned table, NOT a fully matched reimplementation',
        raw_data_and_predictions=references,
        removed_from_paper=['release-layout comparison','shared-response speedup ablation'],
        retained_computational_requirement='Common forward response in4.1 remains necessary for a controlled solver comparison')
    (output/'data_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if output!=OUTPUT:
        (output/'README.md').write_text(
            '# Expanded manuscript data\n\n'
            f'Complete main cohort:120 cases; supplementary cohort:{planned} cases; '
            f'three-method rows:{len(combined)}. All planned cases are retained.\n\n'
            'combined_case_metrics.csv contains every case with its source study. '
            'main_* and supplement_* tables keep the cohorts separate. '
            'Calibration and controlled-release tables are unchanged in scope. '
            'data_manifest.json links raw inputs, results and optimization traces with SHA256.\n\n'
            'The Word summary is generated only after all supplementary fits finish. '
            'Generation alone does not confirm visual layout review. '
            f'Check ../{supplement.name}/run_status.json for completion.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='raw_data_and_predictions'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
