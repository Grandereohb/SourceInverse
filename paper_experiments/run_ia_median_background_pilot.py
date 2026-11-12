"""IA-only pre-event median background pilot; measurements/NLS remain unchanged."""

import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,AUDIT,STUDY,read,save,JointBounds
from run_ia_background_pilot import pre_event
from metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models

DEST=STUDY/'ia_median_background_pilot_20261028'


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    parent=read(OUT/'protocol.json')
    cases=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources=[Path(__file__),ROOT/'paper_experiments/run_ia_background_pilot.py',
             ROOT/'paper_experiments/paper_q_model_selection.py']
    protocol=dict(cases=cases,audit_cases=parent['event_ids'],
        selection='Same two public panel events, frozen before fitting; no outcome-based case selection',
        background='Station-wise median of the SAME independent pre-event30min samples; same prior3h fallback',
        purpose='Test background center bias; does not claim the prior quantile assumption was a coding error',
        calibration='Only pre-event methane, no event/source-location/rate truth; no search over quantiles',
        measurements='Raw concentration/wind/stations unchanged, all finite event rows retained and signed residuals kept',
        fit='Original frozen mean-wind response, Q BIC1/3/5, original starts/bounds/cap, nfev250',
        nls='Unchanged original20th-quantile inputs and stored results, not executed',
        interpretation='Full-method preprocessing comparison, not isolated variable-projection evidence',
        orders=parent['numerical_orders'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve prior median-background pilot')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    table=raw.pivot_table(index='timestamp',columns='name',values='methane',aggfunc='mean')
    audit=[]
    for identifier in protocol['audit_cases']:
        original=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        pre=pre_event(identifier,table)
        quantile=np.nanquantile(pre,.2,axis=0)
        median=np.nanmedian(pre,axis=0)
        np.testing.assert_allclose(quantile,original['baseline'],rtol=1e-12,atol=1e-12)
        shift=median-quantile
        record=dict(background_median_ppm=median,original_background_q20_ppm=quantile,
            shift_ppm=shift,pre_event_finite_counts=np.isfinite(pre).sum(0),
            pre_event_mean_ppm=np.nanmean(pre,axis=0),original_samples_verified=True)
        save(DEST/identifier/'background_calibration.json',record)
        audit.append(dict(event_id=identifier,median_shift_ppm=float(np.median(shift)),
            max_shift_ppm=float(shift.max()),min_pre_samples=int(np.isfinite(pre).sum(0).min())))
    pd.DataFrame(audit).to_csv(DEST/'background_audit.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for identifier in cases:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        baseline=np.array(read(DEST/identifier/'background_calibration.json')['background_median_ppm'])
        mask=np.isfinite(data['concentration'])
        new_data={key:value.copy() for key,value in data.items()}
        new_data['baseline']=baseline
        new_data['concentration']=data['raw_concentration']-baseline
        np.testing.assert_array_equal(np.isfinite(new_data['concentration']),mask)
        np.testing.assert_array_equal(new_data['raw_concentration'],data['raw_concentration'])
        np.testing.assert_array_equal(new_data['wind_xy'],data['wind_xy'])
        output=DEST/identifier/'ia_fit.json'
        if output.exists():
            fit=read(output)
        else:
            print('MEDIAN_BG_START',identifier,flush=True)
            orders=protocol['orders']
            model,xyt,basis,response,observed=configure(new_data,meta['source_height_m'],device='cuda',
                time_order=orders['time'],release_order=orders['release'],history_minutes=orders['history_minutes'])
            initial=read(OUT/'fits'/identifier/'initialization.json')
            objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
            fit=fit_profiled_q_models(objective,[np.array(start) for start in initial['starts']],max_nfev=250)
            fit.update(background_mode='independent_pre_event_median',background_median_ppm=baseline,
                measurement_rows_unchanged=True,nls_called=False)
            save(output,fit)
            np.savez_compressed(DEST/identifier/'ia_input.npz',**new_data)
            del objective,response,model
            gc.collect()
            torch.cuda.empty_cache()
        # True source information is only used after the fixed preprocessing and fit.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_median_background',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],raw_rmse_ppm=float(np.sqrt(item['mse'])),
                converged=item.get('converged'),q_basis_count=item.get('q_basis_count',5)))
        print('MEDIAN_BG_DONE',identifier,rows[-3],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,nls_called=False,production_promoted=False,
        numerical_replay_pending=True,full_goal_achieved=False))


if __name__=='__main__':
    run()
