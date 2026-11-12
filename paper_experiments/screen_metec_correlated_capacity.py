"""Fixed-criterion rescore of existing release models; no fitting or result replacement."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_metec_comparison import ROOT,OUT,AUDIT,read,save
from run_ia_covariance_pilot import noise_table
from metec_point_adapter import xy_from_lon_lat
from paper_correlated_capacity import residual_energy_dof,correlated_capacity_score

DEST=OUT.parent/'metec_correlated_capacity_screen_20261027'


def run():
    protocol=read(OUT/'protocol.json')
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    save(DEST/'protocol.json',dict(version='correlated_energy_capacity_screen_v1',
        cases=protocol['event_ids'],criterion='nu*log(SSE/N)+(4+Q_count)*log(nu)',
        nu='N^2/tr(C^2), unit marginal variance, independent stations, AR1 correlation calibrated before event',
        fallback='Original BIC if nu is at most the smallest model dimension; otherwise exclude models with dimension >=nu',
        status='Approximate model-capacity screening only; fitting objective unchanged; no production switch',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]
    for identifier in protocol['event_ids']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        old=read(OUT/'fits'/identifier/'ia_vpdi.json')
        calibration=noise_table(identifier,raw)
        valid=np.flatnonzero(np.isfinite(data['concentration'].ravel()))
        station_count=data['concentration'].shape[1]
        nu=residual_energy_dof(valid//station_count,valid%station_count,calibration['station_rho'])
        models=old['release_models']
        eligible=[m for m in models if 4+m['q_basis_count']<nu]
        if eligible:
            selected=min(eligible,key=lambda m:correlated_capacity_score(m['sse'],len(valid),nu,4+m['q_basis_count']))
        else:
            selected=min(models,key=lambda m:m['bic'])
        save(DEST/identifier/'screen.json',dict(energy_dof=nu,observations=len(valid),calibration=calibration,
            selected=selected,fallback=not bool(eligible),rescoring_only=True,
            warning='Unselected old fits omit diffusion and prediction; not a deployable fitted result'))
        # True coordinates are read only after the capacity decision.
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        nls=read(OUT/'fits'/identifier/'joint_nls_trf.json')
        for name,fit in [('correlated_capacity_screen',selected),('original_ia',old),('joint_nls',nls)]:
            rows.append(dict(event_id=identifier,method=name,energy_dof=nu,observations=len(valid),
                q_basis_count=fit.get('q_basis_count',5),
                localization_error_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy))))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    summary=frame.groupby('method').localization_error_m.agg(['median','mean','count']).reset_index()
    summary.to_csv(DEST/'summary.csv',index=False,encoding='utf-8-sig')
    print(summary.to_string(index=False),flush=True)
    save(DEST/'status.json',dict(screen_complete=True,fits_performed=False,full_goal_achieved=False,
        note='Full cohort rescoring of existing models, not independent validation or final uncertainty evidence'))


if __name__=='__main__':
    run()
