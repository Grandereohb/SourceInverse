"""Bounded-memory decomposition of existing METEC numerical warnings; no refitting."""

import argparse
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,METHODS,read,save
from ia_metec_point_adapter import configure,window_mean_release,xy_from_lon_lat

DEST=OUT.parent/'metec_numerical_audit_20261027'
ORDERS=[(16,32,20),(32,16,20),(32,32,20),(32,32,40),(64,64,40)]


def run(device,batch_minutes):
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    cases=[]
    selected={}
    for identifier in parent['event_ids']:
        replay=read(OUT/'replay'/(identifier+'.json'))
        if not any(not value['passed'] for value in replay.values()):
            continue
        cases.append(identifier)
        worst=max(replay,key=lambda key:replay[key]['relative_l2'])
        selected[identifier]=list(dict.fromkeys(['ia_vpdi_0','joint_nls_trf_0',worst]))
    source_paths=[Path(__file__),ROOT/'paper_experiments/metec_point_adapter.py',
        ROOT/'paper_experiments/paper_physical_beam_adapter.py']
    protocol=dict(version='numerical_error_decomposition_v1',cases=cases,selected=selected,
        selection='All events failing the frozen1% replay; IA/NLS and largest numerical warning per event',
        orders=ORDERS,batch_minutes=batch_minutes,
        batching='All observed minutes concatenated in original order; full wind/Q history retained per batch',
        method='Saved parameters and Q; no refit or truth-based selection',device=device,
        parent_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    # JSON roundtrip normalizes tuples for checkpoint comparison.
    import json
    protocol=json.loads(json.dumps(protocol))
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve earlier numerical audit; use a new version')
    save(DEST/'protocol.json',protocol)
    summary=[]
    for identifier in cases:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        fits={}
        for key in selected[identifier]:
            method,repeat=key.rsplit('_',1)
            saved=read(OUT/'fits'/identifier/(method+'.json'))
            fits[key]=saved.get('replicates',[saved])[int(repeat)]
        initial=read(OUT/'fits'/identifier/'initialization.json')
        save(DEST/identifier/'fit_sources.json',{key:dict(
            source_norm=fit['source_norm'],diffusion_norm=fit['diffusion_norm'],
            coefficients=fit['coefficients']) for key,fit in fits.items()})
        predictions={}
        for time_order,release_order,history in ORDERS:
            label=f't{time_order}_r{release_order}_h{history}'
            path=DEST/identifier/(label+'.npz')
            if path.exists():
                predictions[label]=dict(np.load(path))
                continue
            print('QUADRATURE_START',identifier,label,flush=True)
            begun=time.perf_counter()
            parts={key:[] for key in fits}
            for first in range(0,len(data['concentration']),batch_minutes):
                query=np.arange(first,min(first+batch_minutes,len(data['concentration'])))
                model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device=device,
                    time_order=time_order,release_order=release_order,history_minutes=history,
                    observation_minutes=query)
                for key,fit in fits.items():
                    parallel,perpendicular=fit['diffusion_norm']
                    theta=model.xs.new_tensor([*fit['source_norm'],perpendicular,parallel-perpendicular])
                    values=(response(model,0.,xyt,basis,physical_parameters=theta)@
                        model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                    if not np.isfinite(values).all():
                        raise FloatingPointError('Nonfinite numerical replay')
                    parts[key].append(values)
                del model,response,basis,xyt
            predictions[label]={key:np.concatenate(values) for key,values in parts.items()}
            np.savez_compressed(path,**predictions[label])
            save(DEST/identifier/(label+'_timing.json'),dict(seconds=time.perf_counter()-begun))
            gc.collect()
            if device=='cuda':
                torch.cuda.empty_cache()
            print('QUADRATURE_DONE',identifier,label,round(time.perf_counter()-begun,1),flush=True)
        # Truth is evaluation-only, after every saved-parameter replay has finished.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        observed=data['concentration'].ravel()
        observed=observed[np.isfinite(observed)]
        for key,fit in fits.items():
            reference=predictions['t64_r64_h40'][key]
            norm=max(float(np.linalg.norm(reference)),1e-30)
            def relative(left,right):
                return float(np.linalg.norm(left-right)/norm)
            summary.append(dict(event_id=identifier,method_repeat=key,
                original_to_reference=relative(np.asarray(fit['prediction']),reference),
                release16_to32=relative(np.asarray(fit['prediction']),predictions['t16_r32_h20'][key]),
                time16_to32=relative(np.asarray(fit['prediction']),predictions['t32_r16_h20'][key]),
                history20_to40=relative(predictions['t32_r32_h20'][key],predictions['t32_r32_h40'][key]),
                orders32_to64=relative(predictions['t32_r32_h40'][key],reference),
                reference_rmse=float(np.sqrt(np.mean((reference-observed)**2))),
                original_rmse=float(np.sqrt(fit['mse'])),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],
                location_error_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy)),
                q_cap_kg_s=max(initial['bounds']['upper'][4:])))
    frame=pd.DataFrame(summary)
    frame.to_csv(DEST/'decomposition.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(complete=True,refit_performed=False,full_goal_achieved=False,
        reference='64/64 orders,40min history; a convergence comparison, not exact physical truth',
        max_orders32_to64=float(frame.orders32_to64.max()),
        max_history_change=float(frame.history20_to40.max())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--batch-minutes',default=8,type=int)
    args=parser.parse_args()
    if args.batch_minutes<1:
        parser.error('Positive batch size required')
    run(args.device,args.batch_minutes)
