"""Read-only full-cohort audit of saved IA amplitude/transport compensation.

No source is refitted, no NLS is executed, and no diagnostic selects a source.
"""

import gc
import hashlib
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,STUDY,read,save,load_input
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=STUDY/'ia_compensation_audit_20261028'


def surviving_source_fraction(matrix,derivative,coefficients,free_q):
    physical=np.einsum('nkd,k->nd',derivative,coefficients)
    source=physical[:,:2]
    nuisance=np.column_stack((physical[:,2:],matrix[:,free_q]))
    norms=np.linalg.norm(nuisance,axis=0)
    nuisance=nuisance[:,norms>0]/norms[norms>0]
    if nuisance.shape[1]:
        u,s,_=np.linalg.svd(nuisance,full_matrices=False)
        independent=s>1e-10*s[0]
        remaining=source-u[:,independent]@(u[:,independent].T@source)
    else:
        remaining=source.copy()
    norm=np.linalg.norm(source)
    return float(np.linalg.norm(remaining)/norm) if norm>0 else None


def run():
    torch.set_num_threads(1)
    DEST.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/'protocol.json')
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Frozen NLS path changed: '+name)
    rows=[]
    for identifier in protocol['event_ids']:
        fit=read(OUT/'fits'/identifier/'ia_vpdi.json')
        initial=read(OUT/'fits'/identifier/'initialization.json')
        data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',protocol['numerical_orders'])
        theta=np.r_[fit['source_norm'],fit['diffusion_norm'][1],
                    fit['diffusion_norm'][0]-fit['diffusion_norm'][1]]
        matrix,derivative=response.matrix_and_jacobian(model,0.,xyt,basis,model.xs.new_tensor(theta))
        # The selected Q is represented in the common five-hat basis. Use the
        # actual selected capacity for nuisance directions, not five fictitious Qs.
        from q_capacity import nested_hat_transform
        mapping=nested_hat_transform(5,fit['q_basis_count'])
        matrix=matrix.cpu().numpy()@mapping
        derivative=np.einsum('nkd,kj->njd',derivative.cpu().numpy(),mapping)
        a=np.asarray(fit['native_coefficients'])
        prediction=matrix@a
        np.testing.assert_allclose(prediction,fit['prediction'],rtol=1e-8,atol=1e-10)
        lower=np.asarray(initial['bounds']['lower'])[:4]
        upper=np.asarray(initial['bounds']['upper'])[:4]
        qcap=initial['bounds']['upper'][4]
        free=(a>1e-10*max(np.max(a),1e-30))&(a<qcap*(1-1e-8))
        fraction=surviving_source_fraction(matrix,derivative,a,free)
        record=dict(event_id=identifier,q_basis_count=fit['q_basis_count'],
            mean_q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
            max_q_kg_h=float(np.max(a)*3600),q_cap_kg_h=qcap*3600,
            q_cap_fraction=float(np.max(a)/qcap),
            rms_prediction_gain_per_q=float(np.linalg.norm(prediction)/max(np.linalg.norm(a)*np.sqrt(len(observed)),1e-300)),
            surviving_source_gradient_fraction=fraction,
            source_edge_fraction=float(np.min(np.minimum(theta[:2]-lower[:2],upper[:2]-theta[:2])/(upper[:2]-lower[:2]))),
            perp_diffusion_m2_s=theta[2]/.006,parallel_diffusion_m2_s=(theta[2]+theta[3])/.006,
            sse=fit['sse'],raw_rmse=float(np.sqrt(np.mean((prediction-observed)**2))))
        # Truth is attached only after computing all label-free diagnostics.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        record.update(localization_error_m=float(np.linalg.norm(theta[:2]*100-xy)),
            true_q_kg_h=truth['mean_q_kg_h'])
        save(DEST/'events'/(identifier+'.json'),record)
        rows.append(record)
        print('COMPENSATION_AUDIT',identifier,flush=True)
        del response,model,matrix,derivative
        gc.collect()
        torch.cuda.empty_cache()
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'diagnostics.csv',index=False,encoding='utf-8-sig')
    save(DEST/'summary.json',dict(events=len(rows),fits_performed=False,nls_called=False,
        source_estimates_changed=False,
        observations='All saved valid observations; no threshold or case selection',
        source_gradient_fraction='Fraction of local source derivative remaining after projecting out D and free Q directions; not calibrated confidence',
        tangent_note='Diffusion derivatives are treated as unrestricted nuisance directions, including at parameter bounds; conservative local description, not a constrained confidence calculation',
        warning='Local compensation is a diagnostic, not evidence of causal error or a replacement source selector',
        median_remaining_source_fraction=float(frame.surviving_source_gradient_fraction.median()),
        max_mean_q_kg_h=float(frame.mean_q_kg_h.max()),
        q_cap_bound_count=int((frame.q_cap_fraction>=1-1e-6).sum()),
        baseline_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()))


if __name__=='__main__':
    run()
