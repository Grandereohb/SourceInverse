"""Application figure: observed geometry, method disagreement and peak fit.

The three input-preselected examples include the retained refinement failure.
No source-location truth, industrial accuracy metric or cohort statistic exists.
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_paper_formal import export

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pinn_source'))
from geo_utils import latlon_to_xy


def plot():
    study=ROOT/'result/paper_experiments_20261017/industrial_representative_application_v1'
    finished=json.loads((study/'APPLICATION_ATTEMPTS_FINISHED.json').read_text(encoding='utf-8'))
    plan=json.loads((study/'representative_plan.json').read_text(encoding='utf-8'))
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    preparation=ROOT/protocol['preparation'].replace('\\','/')
    selected=plan['selected_case_ids']
    if set(finished['states'])!=set(selected) or finished['all_eligible_cohort_complete']:
        raise ValueError('Bounded application attempts, including failures, must be recorded')
    cases=[]
    for case in selected:
        with np.load(preparation/case/'input.npz') as loaded:
            arrays={key:loaded[key] for key in loaded.files}
        record_path=study/case/'result.json'
        record=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else None
        if finished['states'][case]=='completed' and (record is None or not all(
                v['numerical_gate_passed'] for v in record['methods'].values())):
            raise ValueError('A completed example requires both numerically checked methods')
        cases.append((case,arrays,record))
    lon0=float(np.mean(np.concatenate([a['longitude'] for _,a,_ in cases])))
    lat0=float(np.mean(np.concatenate([a['latitude'] for _,a,_ in cases])))
    positions=[]
    for _,a,r in cases:
        x,y=latlon_to_xy(a['longitude'],a['latitude'],lon0,lat0)
        positions.extend(zip(x/1000.,y/1000.))
        if r:
            for value in r['methods'].values():
                x,y=latlon_to_xy(value['longitude'],value['latitude'],lon0,lat0)
                positions.append((x/1000.,y/1000.))
    positions=np.asarray(positions)
    low,high=positions.min(axis=0)-1.,positions.max(axis=0)+1.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.5))
    fig.subplots_adjust(left=.085,right=.985,top=.83,bottom=.16,wspace=.52,hspace=.58)
    names=('Single-site dominated','Multi-site response','Changing wind')
    methods=(('ia_vpdi','#197A9A','^','IA-VPDI'),
             ('matched_joint_adam','#55585B','s','Matched joint Adam'))
    rows=[]
    for col,(case,arrays,record) in enumerate(cases):
        mask=arrays['observation_mask']
        data=arrays['residual_concentration']
        peak_time,peak_site=np.unravel_index(np.nanargmax(np.where(mask,data,np.nan)),data.shape)
        x,y=latlon_to_xy(arrays['longitude'],arrays['latitude'],lon0,lat0)
        ax=axes[0,col]
        ax.scatter(x/1000.,y/1000.,s=13,facecolors='white',edgecolors='#55585B',linewidths=.7)
        ax.scatter(x[peak_site]/1000.,y[peak_site]/1000.,s=24,marker='D',color='#BC8625')
        if record:
            for key,color,marker,label in methods:
                value=record['methods'][key]
                sx,sy=latlon_to_xy(value['longitude'],value['latitude'],lon0,lat0)
                ax.scatter(sx/1000.,sy/1000.,s=38 if key=='ia_vpdi' else 50,marker=marker,
                           facecolors=color if key=='ia_vpdi' else 'none',edgecolors=color,linewidths=1.)
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
        ax.set_xlabel('East distance (km)')
        ax.set_ylabel('North distance (km)')
        ax.set_title(case+'\n'+names[col],fontsize=8,pad=9)
        ax.text(-.24,1.10,'abc'[col],transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax=axes[1,col]
        hours=np.arange(len(data))
        ax.plot(hours,np.where(mask[:,peak_site],data[:,peak_site],np.nan),
                color='#BC8625',lw=1.1,marker='.',markersize=3)
        if record:
            scale=record['metadata']['concentration_scale']
            for key,color,marker,label in methods:
                full=np.full(mask.shape,np.nan)
                full[mask]=np.asarray(record['methods'][key]['prediction_normalized'])*scale
                ax.plot(hours,full[:,peak_site],color=color,lw=1.1,
                        linestyle='-' if key=='ia_vpdi' else '--')
                for t,observation,prediction in zip(hours,data[:,peak_site],full[:,peak_site]):
                    rows.append(dict(case_id=case,method=key,hour=int(t),
                        peak_station=arrays['station_names'][peak_site],
                        observed_residual=observation,predicted_residual=prediction))
        else:
            ax.set_title('Observed only; refinement stopped',fontsize=7,pad=8,color='#55585B')
            for t,observation in zip(hours,data[:,peak_site]):
                rows.append(dict(case_id=case,method='observed_only',hour=int(t),
                    peak_station=arrays['station_names'][peak_site],
                    observed_residual=observation,predicted_residual=None))
        ax.set_xlabel('Hours from window start')
        ax.set_ylabel('Peak excess (ug/m3)')
        ax.text(-.24,1.10,'def'[col],transform=ax.transAxes,fontweight='bold',fontsize=9)
    handles=[Line2D([],[],marker='D',color='#BC8625',markersize=4,lw=1,label='Peak-site observation')]
    handles.extend(Line2D([],[],marker=m,color=c,markersize=4,lw=1,
                          linestyle='-' if k=='ia_vpdi' else '--',label=l) for k,c,m,l in methods)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.985),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.025,'Input-preselected examples; no source-location truth. The stopped example is not excluded.',
             ha='center',fontsize=7)
    fig.canvas.draw()
    bounds=axes[0,0].get_window_extent()
    ratio=bounds.width/bounds.height
    span_x=max(high[0]-low[0],(high[1]-low[1])*ratio)
    span_y=span_x/ratio
    center=(low+high)/2
    for ax in axes[0]:
        ax.set_xlim(center[0]-span_x/2,center[0]+span_x/2)
        ax.set_ylim(center[1]-span_y/2,center[1]+span_y/2)
        ax.set_aspect('equal',adjustable='box')
    output=study/'analysis/figures'
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'peak_site_source_data.csv',index=False)
    export(fig,output/'industrial_application')


if __name__=='__main__':
    plot()
