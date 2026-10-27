"""All-event error comparison and the chronologically first complete METEC release."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_metec_comparison import OUT,METHODS,read,save

SKILL=Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

LABELS=['IA-VPDI','Joint NLS','Standard PSO','Joint Adam']
COLORS=['#167C80','#65578C','#737373','#AD763E']


def export(fig,name):
    folder=OUT/'figures'
    folder.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,json_out=folder/(name+'.alignment.json'),strict=True)
    fig.savefig(folder/(name+'.png'),dpi=600)
    fig.savefig(folder/(name+'.svg'))
    fig.savefig(folder/(name+'.pdf'))
    plt.close(fig)


def plot():
    if not read(OUT/'analysis/summary.json')['final']:
        raise ValueError('Complete cohort required')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    case=pd.read_csv(OUT/'analysis/case_metrics.csv',dtype={'event_id':str})
    raw=pd.read_csv(OUT/'analysis/replicate_metrics.csv',dtype={'event_id':str})
    ids=read(OUT/'protocol.json')['event_ids']
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
    fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.8,wspace=.32)
    for j,(metric,label,mult) in enumerate([('localization_error_m','Location error (m)',1),
        ('q_relative_error','Emission rate error (%)',100)]):
        ax=axes[j]
        for method,name,color in zip(METHODS,LABELS,COLORS):
            values=case[case.method==method].set_index('event_id').loc[ids,metric].to_numpy()*mult
            ax.plot(np.arange(1,18),values,label=name,color=color,
                linestyle='--' if method=='joint_nls_trf' else '-',
                marker='s' if method=='joint_nls_trf' else ('o' if method=='ia_vpdi' else '.'),
                markerfacecolor='none' if method=='joint_nls_trf' else color,
                markersize=4 if method=='joint_nls_trf' else 3,linewidth=1.1,
                zorder=6 if method=='joint_nls_trf' else (5 if method=='ia_vpdi' else 3))
            if method=='standard_gbest_pso':
                spread=raw[raw.method==method].groupby('event_id')[metric].agg(['min','max']).loc[ids]*mult
                ax.fill_between(np.arange(1,18),spread['min'],spread['max'],color=color,alpha=.1,linewidth=0)
        ax.set(xlabel='Release number in chronological order',ylabel=label,xticks=[1,5,9,13,17])
        ax.set_yscale('symlog',linthresh=1 if j==0 else 10)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.2)
        ax.set_title(chr(97+j),loc='left',fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,.99),fontsize=8)
    export(fig,'all_releases')
    identifier=ids[0]
    data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
    metadata=read(OUT/'inputs'/identifier/'metadata.json')
    values=raw[(raw.event_id==identifier)&((raw.method!='standard_gbest_pso')|(raw.repeat==0))]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.8,wspace=.32)
    ax=axes[0]
    ax.scatter(*data['stations_xy'].T,s=14,color='#B5B5B5',label='Sensors')
    ax.scatter(values.true_x_m.iloc[0],values.true_y_m.iloc[0],marker='*',s=75,color='black',label='True source',zorder=8)
    for method,name,color in zip(METHODS,LABELS,COLORS):
        row=values[values.method==method].iloc[0]
        ax.scatter(row.source_x_m,row.source_y_m,s=30,facecolors='none',edgecolors=color,
            marker='o' if method=='ia_vpdi' else 's',label=name,zorder=6)
    ax.set(xlabel='East (m)',ylabel='North (m)')
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_title('a',loc='left',fontweight='bold')
    ax=axes[1]
    station=int(np.nanargmax(np.nanmax(data['concentration'],axis=0)))
    minutes=np.arange(len(data['concentration']))
    ax.plot(minutes/60,data['concentration'][:,station],color='black',alpha=.5,linewidth=.6,label='Observed')
    mask=np.isfinite(data['concentration'].ravel())
    for method,name,color in zip(METHODS,LABELS,COLORS):
        fit=read(OUT/'fits'/identifier/(method+'.json'))
        if method=='standard_gbest_pso':
            fit=fit['replicates'][0]
        field=np.full(data['concentration'].size,np.nan)
        field[mask]=fit['prediction']
        ax.plot(minutes/60,field.reshape(data['concentration'].shape)[:,station],color=color,linewidth=.8,
            linestyle='--' if method=='joint_nls_trf' else '-',
            zorder=6 if method=='joint_nls_trf' else (5 if method=='ia_vpdi' else 3))
    ax.set(xlabel='Time from window start (h)',ylabel='Methane increment (ppm)')
    ax.set_title('b',loc='left',fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    h,l=axes[1].get_legend_handles_labels()
    fig.legend(handles+h,labels+l,loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,fontsize=7)
    export(fig,'first_release')
    save(OUT/'figures/captions.json',dict(all_releases='All17 releases retained. PSO is median of3 fixed seeds per event; grey band is min-max technical seed spread, not confidence interval. No seed-count inflation. Symlog axes retain zero.',
        first_release=f'First chronological complete release {identifier}; station {metadata["station_names"][station]}. PSO uses fixed seed1729, not best seed. Concentrations are background-subtracted. Overlap means similar solutions.'))


if __name__=='__main__':
    plot()
