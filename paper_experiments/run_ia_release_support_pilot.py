"""Four prespecified synthetic geometries, adaptive release support; no winner filtering."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_paper_updated_evidence import STUDY,read,save,synthetic_setup,prior
from paper_release_support import fit_release_support

OUT=STUDY/'ia_release_support_screened_pilot_20261027'


def run(device):
    torch.set_num_threads(1)
    selected=[]
    networks=set()
    for entry in read(STUDY/'new_comparison_v4/protocol.json')['cases']:
        meta=entry['case']
        if meta['cohort']=='nominal' and meta['network'] not in networks:
            selected.append(entry)
            networks.add(meta['network'])
    files=[Path(__file__),Path(__file__).with_name('paper_release_support.py')]
    protocol=dict(version='release_support_screened_pilot_v2',cases=selected,
        selection='First nominal case per geometry, frozen protocol order, development only',
        model='Same five-hat physical response; all31 nonempty supports plus constant and nested3',
        screening='All33 scored using common starts plus internally profiled full5; refine best support per size1..4 plus constant/nested3/full5',
        score='Gaussian unknown-variance BIC; uniform discrete model prior; no new empirical penalty',
        q_elimination=True,baseline='Existing NLS and IA unchanged; no true release labels used for fitting',
        max_nfev=100,starts='Original shared two observation-only starts',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=protocol:
        raise ValueError('Changed support pilot; use a different version')
    save(OUT/'protocol.json',protocol)
    rows=[]
    for entry in selected:
        identifier=entry['case']['case_id']
        old,data,objective,starts=synthetic_setup(entry,device)
        path=OUT/identifier/'result.json'
        if path.exists():
            fit=read(path)
        else:
            print('SUPPORT_START',identifier,flush=True)
            fit=fit_release_support(objective,starts,protocol['max_nfev'])
            # Evaluation truth is accessed only after the optimization has completed.
            fit.update(prior.formal.metrics(old['metadata'],data,fit['source_norm'],
                np.array(fit['coefficients']),np.array(fit['prediction'])))
            save(path,fit)
            print('SUPPORT_DONE',identifier,fit['layout_id'],fit['localization_error_m'],flush=True)
        previous=read(STUDY/'ia_release_capacity_full_v1'/identifier/'result.json')['methods']['ia_vpdi_release_bic']
        for name,item in [('ia_release_support',fit),('ia_vpdi',previous),
            ('joint_nls_trf',old['methods']['joint_nls_trf'])]:
            rows.append(dict(case_id=identifier,method=name,localization_error_m=item['localization_error_m'],
                q_relative_l2=item['q_relative_l2'],mse=item.get('mse',item.get('observed_mse')),
                layout=item.get('layout_id'),q_basis_count=item.get('q_basis_count',5)))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    save(OUT/'status.json',dict(pilot_complete=True,full_goal_achieved=False,
        note='Four development cases do not prove full synthetic/public superiority'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args().device)
