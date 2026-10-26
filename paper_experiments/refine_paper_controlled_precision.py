"""Versioned32-order rerun selected ONLY by failed1% numerical replay gates.

Same grid, seeds, two starts,500 updates and observations; no warm start at
the old fitted solution. Other fitted methods are copied byte-for-byte.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import fit_paper_controlled_release as fit

physical,base=fit.physical,fit.base


def main():
    torch.set_num_threads(1)
    old=fit.ROOT/'controlled_release_fit_v1'
    output=fit.ROOT/'controlled_release_fit_v2_precision'
    if output.exists():
        raise FileExistsError('Preserve versioned numerical evidence; do not overwrite')
    repairs=[]
    for path in sorted(old.glob('*/height_*/result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        for method,value in result['methods'].items():
            if method.startswith('transient_') and not value['replay_gate_passed']:
                repairs.append(dict(event=path.parent.parent.name,height=result['height_m'],method=method))
    if len(repairs)!=2 or any(v['method']!='transient_hat5' for v in repairs):
        raise ValueError('This rescue protocol covers exactly the two recorded hat5 quadrature failures')
    shutil.copytree(old,output,ignore=shutil.ignore_patterns('analysis','RUN_FINISHED.json'))
    shutil.copyfile(old/'protocol.json',output/'original_protocol.json')
    protocol=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(numerical_repairs=repairs,repair_time_order=32,repair_release_order=32,
        repair_replay_order=64,repair_selection='Failed numerical gate only; no truth or fit-error selection',
        repair_optimizer='Fresh full coarse search and two500-step starts, NOT extra warm-start optimization',
        original_result_sha256={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in old.glob('*/height_*/result.json')})
    protocol['sha256'][str(Path(__file__).relative_to(base.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base.save_json(output/'protocol.json',protocol)
    snapshot=output/'source_snapshot'/Path(__file__).relative_to(base.ROOT)
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,snapshot)
    for repair in repairs:
        event,height=repair['event'],repair['height']
        destination=output/event/f'height_{height:.1f}'
        started=time.perf_counter()
        model,xyt,basis,response,observed,observations=physical.configure(height,device='cuda',dtype=torch.float32,
            time_order=32,release_order=32,beam_solver='analytic',data_dir=fit.DATA/event)
        candidates=[]
        with patch.object(base.ia,'build_response_matrix',response):
            for dq in (3.25,6.5,13.):
                for ratio in (1.,3.):
                    dp=dq*ratio+(1e-6 if ratio==1. else 0.)
                    for x in np.linspace(0.,1.1,7):
                        for y in np.linspace(0.,1.1,7):
                            candidates.append(base.ia.score_all_data_candidate(model,0.,xyt,observed,basis,
                                x,y,dp*.006,dq*.006))
            candidates.sort(key=lambda c:c.profile_score)
            first=candidates[0]
            second=next(c for c in candidates if np.hypot(c.source_x-first.source_x,c.source_y-first.source_y)>=.2)
            runs=[base.ia.refine_profiled_candidate(model,0.,xyt,observed,basis,start,
                (0.,1.1),(0.,1.1),steps=500,lr=.02,patience=501,min_delta=0.,run_id=i)
                for i,start in enumerate((first,second))]
        if not all(r.success for r in runs):
            raise RuntimeError('Higher-precision refinement failed; preserve original failures')
        best=min((r.best for r in runs),key=lambda c:c.profile_score)
        fine_model,z,phi,replay,_,_=physical.configure(height,device='cpu',dtype=torch.float64,
            time_order=64,release_order=64,beam_solver='analytic',data_dir=fit.DATA/event)
        with base.ia.temporary_source_and_diffusion(fine_model,best.source_x,best.source_y,
                best.diffusion_parallel_norm,best.diffusion_perp_norm):
            prediction=replay(fine_model,0.,z,phi).numpy()@best.coefficients
        relative=float(np.linalg.norm(prediction-best.prediction)/max(np.linalg.norm(prediction),np.finfo(float).tiny))
        method=dict(best=asdict(best),runs=[asdict(r) for r in runs],
            refinement_and_replay_seconds=time.perf_counter()-started,
            timing_scope='Includes independent32-order coarse search; unlike original refinement-only timing',
            source_m=[best.source_x*100.,best.source_y*100.],
            diffusion_m2_s=[best.diffusion_parallel_norm/.006,best.diffusion_perp_norm/.006],
            mean_q_kg_s=fit.mean_hat_q(best.coefficients,len(observations)//7),
            replay_relative_difference=relative,replay_gate_passed=relative<=.01,
            optimizer_convergence='unverified fixed budget',integration_order=32,replay_order=64)
        base.save_json(destination/'transient_hat5.json',method)
        frame=observations[['time','beam','residual_ppm']].copy()
        frame['predicted_ppm']=best.prediction
        frame['fine_replay_ppm']=prediction
        frame.to_csv(destination/'transient_hat5_predictions.csv',index=False)
        pd.DataFrame(dict(minutes=model.recurrent_integration_times.cpu().numpy(),
            q_kg_s=best.q_integration,q_kg_h=best.q_integration*3600.)).to_csv(
                destination/'transient_hat5_q.csv',index=False)
        result=json.loads((destination/'result.json').read_text(encoding='utf-8'))
        result['methods']['transient_hat5']=method
        result['precision_repair']=repair
        base.save_json(destination/'result.json',result)
        print('PRECISION_REPAIR',event,height,'relative',relative,'seconds',time.perf_counter()-started,flush=True)
        if relative>.01:
            base.save_json(output/'STOPPED.json',dict(reason='32-vs64 numerical gate still failed',repair=repair))
            raise RuntimeError('Do not relax numerical gate or tune to truth')
    base.save_json(output/'RUN_FINISHED.json',dict(events=8,heights=2,repaired_methods=2,
        original_v1_preserved=True,status='fits complete; post-fit evaluation and copy-integrity audit required'))


if __name__=='__main__':
    main()
