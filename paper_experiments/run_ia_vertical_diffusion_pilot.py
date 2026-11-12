"""IA-only physical diffusivity pilot with an untouched archived NLS baseline."""

import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,STUDY,read,save,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from ia_vertical_diffusion import fit_vertical_diffusion

PREVIOUS=STUDY/'ia_location_capacity_pilot_20261028'
DEST=STUDY/'ia_vertical_diffusion_pilot_20261028'


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    previous=read(PREVIOUS/'protocol.json')
    files=[Path(__file__),ROOT/'paper_experiments/ia_vertical_diffusion.py',
           ROOT/'paper_experiments/ia_metec_point_adapter.py']
    protocol=dict(event_ids=previous['public'][:2],
        selection='First two public events of the prior fixed pilot panel, before new fitting',
        physical_model='Independent Dz vs tied Dz=Dperp; same time-varying mean wind and reflected Gaussian kernel',
        vertical_bounds_m2_s=[.1,60.],q_capacity='Fixed to existing IA selected Q capacity',
        criterion='Working Gaussian BIC with one extra physical parameter; pure SSE and all observations',
        starts='Original common starts plus previous IA, each initialized Dz=Dperp',
        nls='Frozen historical results only; no calls',max_nfev=250,
        numerical_orders=dict(time=16,release=16,history_minutes=20),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve old protocol before changing this pilot')
    save(DEST/'protocol.json',protocol)
    rows=[]
    for identifier in protocol['event_ids']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        model,xyt,basis,response,observed=configure(data,meta['source_height_m'],
            device='cuda',time_order=16,release_order=16,history_minutes=20)
        initial=read(OUT/'fits'/identifier/'initialization.json')
        objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
        output=DEST/identifier/'result.json'
        if output.exists():
            result=read(output)
        else:
            old=read(PREVIOUS/'public'/identifier/'result.json')
            reference=next(m for m in old['release_models'] if m['q_basis_count']==old['q_basis_count'])
            print('VERTICAL_START',identifier,flush=True)
            result=fit_vertical_diffusion(objective,[np.array(s) for s in initial['starts']],reference)
            save(output,result)
        # Source/rate truth is used only here, after parameter and model selection.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_vertical_selected',result['selected']),
                            ('ia_vertical_independent',result['independent']),
                            ('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            prediction=np.asarray(item.get('prediction',[]))
            rows.append(dict(case_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                vertical_diffusion_m2_s=item.get('vertical_diffusion_norm',item['diffusion_norm'][1])/.006,
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                selected_mode=result['selected_mode'],sse=item.get('sse'),
                peak_ratio=float(prediction.max()/observed.max()) if prediction.size else None,
                converged=item.get('converged')))
        print('VERTICAL_DONE',identifier,rows[-4],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
        del objective,response,model
        gc.collect()
        torch.cuda.empty_cache()
    save(DEST/'status.json',dict(pilot_complete=True,nls_called=False,
        production_promoted=False,full_goal_achieved=False,
        note='Two development events only; numerical replay and full evaluation not completed'))


if __name__=='__main__':
    run()
