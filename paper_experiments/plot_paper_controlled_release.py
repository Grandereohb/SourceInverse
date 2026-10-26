"""Complete real-release evaluation, both disputed source heights shown."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_paper_formal import export


def plot(analysis):
    status=json.loads((analysis/'analysis_status.json').read_text(encoding='utf-8'))
    if not status['final'] or status['completed_height_jobs']!=16:
        raise ValueError('Require all eight events and both source heights, not pilot or partial fits')
    frame=pd.read_csv(analysis/'case_metrics.csv')
    if len(frame)!=48:
        raise ValueError('Expected three methods per event/height')
    methods=('transient_constant_q','transient_hat5','stationary_constant_q')
    colors=('#197A9A','#55585B','#BC8625')
    labels=('Transient constant Q','Transient (5 hats)','Stationary constant Q')
    events=list(dict.fromkeys(frame.event))
    maximum_location=float(frame.localization_error_m.max())
    maximum_ratio=float((frame.mean_q_kg_h/frame.reference_q_kg_h).max())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.))
    fig.subplots_adjust(left=.1,right=.97,top=.87,bottom=.14,wspace=.3,hspace=.53)
    missing=set()
    for row,height in enumerate((.3,1.6)):
        selected=frame[frame.height_m==height]
        for col in range(2):
            ax=axes[row,col]
            for i,(method,color,label) in enumerate(zip(methods,colors,labels)):
                values=selected[selected.method==method].set_index('event').loc[events]
                metric=values.localization_error_m if col==0 else values.mean_q_kg_h/values.reference_q_kg_h
                if col==1:
                    missing.update(values.index[metric.isna()])
                x=np.arange(len(events))+(i-1)*.16
                ax.plot(x,metric,color=color,linestyle='none',marker=('o','s','^')[i],
                        markersize=4,label=label)
                invalid=~values.quadrature_gate_passed
                if invalid.any():
                    ax.plot(x[invalid],metric[invalid],linestyle='none',marker='x',
                            color='#B73340',markersize=7)
            if col==0:
                ax.set_yscale('symlog',linthresh=.1)
            ax.set_ylim(0.,1.08*(maximum_location if col==0 else maximum_ratio))
            if col==1:
                ax.axhline(1.,color='#8A8A8A',ls='--',lw=.8,zorder=0)
                positions=np.asarray(sorted(events.index(event) for event in missing))
                if len(positions):
                    groups=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
                    for group in groups:
                        ax.text(float(group.mean()),.04,'No weighed\nreference',fontsize=6,
                            ha='center',va='bottom',transform=ax.get_xaxis_transform(),color='#55585B')
            ax.set_xlim(-.5,len(events)-.5)
            ax.set_xticks(np.arange(len(events)),[v.replace('source','S').replace('_rel','/R') for v in events],
                          rotation=40,ha='right',rotation_mode='anchor',fontsize=7)
            ax.set_ylabel('Horizontal location error (m)' if col==0 else 'Estimated / weighed mean Q')
            ax.set_title(f'Source height {height:.1f} m',fontsize=9)
            ax.text(-.14,1.07,'abcd'[row*2+col],transform=ax.transAxes,fontsize=10,fontweight='bold')
    handles,_=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,columnspacing=1.)
    output=analysis/'figures'
    output.mkdir(exist_ok=True)
    export(fig,output/'controlled_release_validation')
    (output/'figure_notes.json').write_text(json.dumps(dict(events=events,heights_m=[.3,1.6],
        observations='All valid prepared beam-minute signed residuals fitted; no station/time withholding',
        rate_missing_events=sorted(missing),rate_reference='Only measured per-event cylinder slopes; no aggregate substitution',
        marker_warning='Red crosses flag failed fixed-parameter quadrature replay; results retained, not silently excluded',
        unit='Eight events but only two source locations and one beam network; not eight independent geometries',
        status='Fixed-budget transient optima unverified; both heights reported without favorable selection'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    plot(parser.parse_args().analysis.resolve())
