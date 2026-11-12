"""IA-only transport/release dimension pilot; frozen NLS results are read-only."""

import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_paper_updated_evidence import STUDY,synthetic_setup,prior
from run_metec_comparison import ROOT,OUT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from ia_transport_capacity import fit_transport_capacity
from metec_point_adapter import xy_from_lon_lat,window_mean_release

PREVIOUS=STUDY/'ia_location_capacity_pilot_20261028'
DEST=STUDY/'ia_transport_capacity_pilot_20261028'


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    previous=read(PREVIOUS/'protocol.json')
    files=[Path(__file__),ROOT/'paper_experiments/ia_transport_capacity.py',
        ROOT/'paper_experiments/paper_location_capacity.py']
    protocol=dict(version='ia_only_transport_capacity_pilot_v1',
        synthetic=previous['synthetic'],public=previous['public'],
        diagnostic_public_event='20220512005',
        cases='Same fixed7-case development panel; no outcome-based case deletion',
        model='Nested isotropic3-parameter and anisotropic4-parameter physical coordinates; each Q1/3/5',
        training='Same frozen forward, all observations, pure SSE, no new loss term',
        selection='Original working Gaussian BIC with correct physical/Q dimension',max_nfev=250,
        reuse='Previous IA anisotropic candidates, verified against the frozen forward; lift scalar fits only if nesting SSE is violated',
        nls='No solver calls or changes; historical result files only',
        timing='Incremental new fits only, excludes cached IA work; not a speed comparison',
        previous_protocol_sha256=hashlib.sha256((PREVIOUS/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Changed IA pilot; preserve the previous version')
    save(DEST/'protocol.json',protocol)
    rows=[]
    for entry in protocol['synthetic']:
        identifier=entry['case']['case_id']
        old,data,objective,starts=synthetic_setup(entry,'cpu')
        output=DEST/'synthetic'/identifier/'result.json'
        if output.exists():
            fit=read(output)
        else:
            print('IA_DIMENSION_START',identifier,flush=True)
            candidates=read(PREVIOUS/'synthetic'/identifier/'result.json')['release_models']
            fit=fit_transport_capacity(objective,starts,candidates,max_nfev=250)
            save(output,fit)
            print('IA_DIMENSION_DONE',identifier,fit['diffusion_mode'],fit['q_basis_count'],flush=True)
        metric=prior.formal.metrics(old['metadata'],data,fit['source_norm'],
            np.array(fit['coefficients']),np.array(fit['prediction']))
        rows.append(dict(dataset='synthetic',case_id=identifier,method='ia_transport_capacity',
            localization_error_m=metric['localization_error_m'],q_relative_l2=metric['q_relative_l2'],
            diffusion_mode=fit['diffusion_mode'],q_basis_count=fit['q_basis_count']))
        reference=read(STUDY/'ia_release_capacity_full_v1'/identifier/'result.json')['methods']['ia_vpdi_release_bic']
        for method,item in [('original_ia',reference),('frozen_nls',old['methods']['joint_nls_trf'])]:
            rows.append(dict(dataset='synthetic',case_id=identifier,method=method,
                localization_error_m=item['localization_error_m'],q_relative_l2=item['q_relative_l2']))
        del objective
        gc.collect()
    for identifier in protocol['public']:
        data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',dict(time=16,release=16,history_minutes=20))
        initial=read(OUT/'fits'/identifier/'initialization.json')
        objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
        output=DEST/'public'/identifier/'result.json'
        if output.exists():
            fit=read(output)
        else:
            print('IA_DIMENSION_START',identifier,flush=True)
            candidates=read(PREVIOUS/'public'/identifier/'result.json')['release_models']
            fit=fit_transport_capacity(objective,[np.array(s) for s in initial['starts']],candidates,max_nfev=250)
            save(output,fit)
            print('IA_DIMENSION_DONE',identifier,fit['diffusion_mode'],fit['q_basis_count'],flush=True)
        # True location/rate is never used to select diffusion or Q dimension.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        comparisons=[('ia_transport_capacity',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]
        for method,item in comparisons:
            rows.append(dict(dataset='public',case_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                diffusion_mode=item.get('diffusion_mode'),q_basis_count=item.get('q_basis_count',5),
                diagnostic_only=identifier==protocol['diagnostic_public_event']))
        del objective,response,model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,nls_optimized=False,full_goal_achieved=False,
        note='Fixed development pilot; not independent validation or production promotion'))


if __name__=='__main__':
    run()
