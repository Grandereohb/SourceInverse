"""Re-evaluate the existing finite candidate pools for the96 nominal cases.

No new locations, optimizer, restart, fit selection or truth-dependent search.
Only solve conditional Q on the old fixed candidates using the new common box.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_paper_formal_experiments as formal
from paper_fast_parcels import CachedMeanResponse

ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def summarize_candidates(surface, selected_source, selected_mse, length_m):
    source = np.asarray(selected_source)
    best = min(float(surface[:, 4].min()), selected_mse)
    records = []
    for tolerance in (.01, .05, .10):
        mask = surface[:, 4] <= best*(1+tolerance)+1e-15
        points = surface[mask, :2]
        selected_included = selected_mse <= best*(1+tolerance)+1e-15
        if selected_included:
            points = np.vstack((points, source))
        points = np.unique(points, axis=0)
        records.append(dict(relative_score_tolerance=tolerance, sampled_locations=len(points),
            farthest_alternative_m=float(np.linalg.norm(points-source, axis=1).max()*length_m),
            sampled_boundary_reached=bool(np.any(np.isclose(np.abs(points), .75, atol=1e-6))),
            selected_within_tolerance=selected_included, selected_mse=selected_mse,
            best_evaluated_mse=best, selected_relative_gap=(selected_mse-best)/max(best, 1e-30),
            interpretation='finite original candidate pool reprofiled under new bounds, not a confidence region; source estimate unchanged'))
    return records


def evaluate(study, device='cpu', limit=None):
    study = Path(study)
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    cases = [item for item in protocol['cases'] if item['case']['cohort'] == 'nominal'
             and item['case']['network'] in ('ideal5', 'jjj6')]
    if len(cases) != 96:
        raise ValueError('Expected the original96 nominal inputs')
    destination = study/'alternative_candidates'
    destination.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    summary, completed = [], 0
    for entry in cases:
        spec = entry['case']
        current = study/spec['case_id']/'result.json'
        if not current.exists():
            raise ValueError('Complete the nominal cohort before candidate diagnostics')
        original = ROOT/entry['parent']/spec['case_id']
        trace = original/'optimization_traces.npz'
        hashes = dict(result_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                      trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        saved = destination/(spec['case_id']+'.npz')
        record = json.loads(current.read_text(encoding='utf-8'))
        if saved.exists():
            with np.load(saved) as archive:
                if any(str(archive[key]) != value for key, value in hashes.items()):
                    raise ValueError('Changed diagnostic input: '+spec['case_id'])
                surface = archive['surface']
        else:
            if limit is not None and completed >= limit:
                break
            case_started = time.perf_counter()
            metadata = record['metadata']
            with np.load(original/'input.npz') as archive:
                stations, observed = archive['stations'].copy(), archive['observed'].copy()
            model, xyt, basis = formal.parcel.configure(metadata, stations, device,
                sigma=metadata['sigma'], wind_interval_mean=True)
            model.double()
            response = CachedMeanResponse(stations=len(stations))
            pool = np.load(trace)['candidate_surface']
            lower, upper = np.asarray(record['bounds']['lower']), np.asarray(record['bounds']['upper'])
            values = []
            with patch.object(formal.base.ia, 'build_response_matrix', response):
                for row in pool:
                    physical = np.array([row[0], row[1], row[3], row[2]-row[3]])
                    if np.any(physical < lower[:4]-1e-10) or np.any(physical > upper[:4]+1e-10):
                        continue
                    fit = formal.base.ia.score_all_data_candidate(model, metadata['sigma'], xyt,
                        observed, basis, *row[:4], q_upper_bound=float(upper[4]))
                    values.append([fit.source_x, fit.source_y, fit.diffusion_parallel_norm,
                        fit.diffusion_perp_norm, fit.profile_score, float(np.max(fit.coefficients))])
            surface = np.asarray(values)
            if not len(surface) or not np.isfinite(surface).all():
                raise ValueError('No finite feasible original candidates: '+spec['case_id'])
            np.savez_compressed(saved, surface=surface, **hashes,
                seconds=time.perf_counter()-case_started, response_calls=response.calls,
                device=device, columns='x,y,D_parallel,D_perp,MSE,max_coefficient')
            completed += 1
            print('DONE_ALTERNATIVES', spec['case_id'], len(surface), flush=True)
        best = record['methods']['ia_vpdi']
        summary.extend(dict(spec, **row) for row in summarize_candidates(surface,
            best['source_norm'], best['observed_mse'], record['metadata']['length_m']))
    analysis = study/'analysis'
    frame = pd.DataFrame(summary)
    frame.to_csv(analysis/'sampled_alternatives.csv', index=False)
    status = dict(final=frame.case_id.nunique() == 96, completed_cases=int(frame.case_id.nunique()),
        planned_cases=96, source_estimates_changed=False, new_candidate_locations_added=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='finite original pool, re-solved Q under common new box; not a global landscape or confidence region')
    (analysis/'alternatives_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'new_comparison_v4')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.study, args.device, args.limit), indent=2))
