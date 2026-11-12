"""IA-only common positive background departures above the frozen baseline."""

import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,STUDY,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from ia_common_background import fit_common_background
from q_capacity import gaussian_bic
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=STUDY/'ia_residual_background_pilot_20261028'


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Frozen NLS changed: '+name)
    parent=read(OUT/'protocol.json')
    cases=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources=[Path(__file__),ROOT/'paper_experiments/ia_common_background.py',ROOT/'paper_experiments/paper_location_capacity.py']
    protocol=dict(cases=cases,
        measurement='Frozen residual observation = positive source + common nonnegative departure from original station baseline',
        calibration='Original q20 baseline is unchanged and acts as a lower reference; negative ambient departures are not modelled',
        differences='Unlike the absolute common-background pilot, it cannot borrow concentration below the original baseline',
        unchanged='Every original signed observation, wind, station, source operator, initial point, bound and Q cap',
        fit='Reduced TRF, nfev250, original2starts, Q1/3/5 with Gaussian working BIC4+K+observed_time_count',
        nls='No runs or modifications; stored original results only',orders=parent['numerical_orders'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous residual-background pilot')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    rows=[]
    for identifier in cases:
        data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',protocol['orders'])
        indices=np.flatnonzero(np.isfinite(data['concentration'].ravel()))
        times=indices//data['concentration'].shape[1]
        initial=read(OUT/'fits'/identifier/'initialization.json')
        objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
        output=DEST/identifier/'ia_fit.json'
        if output.exists():
            fit=read(output)
        else:
            models=[]
            for count in [1,3,5]:
                print('RESIDUAL_BG_START',identifier,count,flush=True)
                function,mapping=capacity_response(objective,count)
                fit=fit_common_background(function,observed,times,[np.array(s)[:4] for s in initial['starts']],
                    objective.bounds.lower[:4],objective.bounds.upper[:4],float(objective.bounds.upper[4]),max_nfev=250)
                theta=fit.pop('theta')
                native=fit['coefficients'].copy()
                fit.update(source_norm=theta[:2],diffusion_norm=[theta[2]+theta[3],theta[2]],
                    native_coefficients=native,coefficients=mapping@native,q_basis_count=count,
                    background_time_indices=np.unique(times),mse=fit['sse']/len(observed),
                    bic=gaussian_bic(fit['sse'],len(observed),4+count+fit['background_groups']))
                models.append(fit)
                print('RESIDUAL_BG_MODEL_DONE',identifier,count,fit['sse'],flush=True)
            fit=dict(min(models,key=lambda item:(item['bic'],item['q_basis_count'])))
            fit['release_models']=[{key:item[key] for key in
                ['q_basis_count','source_norm','diffusion_norm','coefficients','sse','bic','converged','seconds']} for item in models]
            save(output,fit)
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_residual_background',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],raw_rmse_ppm=float(np.sqrt(item['mse'])),
                converged=item.get('converged'),q_basis_count=item.get('q_basis_count',5)))
        print('RESIDUAL_BG_DONE',identifier,rows[-3],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
        del objective,response,model
        gc.collect()
        torch.cuda.empty_cache()
    save(DEST/'status.json',dict(pilot_complete=True,nls_called=False,production_promoted=False,
        numerical_replay_pending=True,full_goal_achieved=False))


if __name__=='__main__':
    run()
