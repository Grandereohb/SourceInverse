"""Prepare one documented single-source release without executing notebooks.

No inversion or mass-rate calibration is performed. Preserve raw wind axes
and source-height disagreement until the physical adapter is validated.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit_paper_controlled_release_data import parse_measurement, parse_measurement_start_label
from paper_observation_operators import beam_quadrature

ROOT=Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def main(event=None,output=None):
    public=ROOT/'public_data'
    event = event or dict(source='source_1',release='rel03',date='20170509',
        archive_directory='20170509/rel03',start='12:49:00',end='13:49:59')
    output=output or ROOT/'controlled_release_preparation'/'source1_rel03'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing prepared observations; use a new version')
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((public/'author_processing_metadata.json').read_text(encoding='utf-8'))
    shared_geometry=public/'author_instrument_geometry.json'
    instruments=(json.loads(shared_geometry.read_text(encoding='utf-8'))['values']
                 if shared_geometry.exists() else metadata['instrument_coordinates_as_printed'])
    day=datetime.strptime(event['date'],'%Y%m%d').date().isoformat()
    start=pd.Timestamp(day+' '+event['start'])
    end=pd.Timestamp(day+' '+event['end'])+pd.Timedelta(microseconds=999999)
    observations,geometry,wind_frames,raw_records=[],[],[],[]
    with zipfile.ZipFile(public/'chilbolton_original.zip') as archive:
        for beam in range(1,8):
            name=f'{event["archive_directory"]}/POS{beam}_{event["release"]}_processed.txt'
            raw=archive.read(name)
            lines,data,columns=parse_measurement(raw.decode('ascii'))
            label=lines[lines.index('Time stamp for record start')+1]
            base=parse_measurement_start_label(label)
            if base.date().isoformat()!=day:
                raise ValueError(f'Header/archive dates disagree: {name}')
            df=pd.DataFrame(data,columns=columns)
            used=['Elapsedtime_sec','Temp_K','Press_Pa','WindX_m/s','WindY_m/s','WindZ_m/s','CH4vmr_ppmv']
            valid=np.isfinite(df[used].to_numpy()).all(1)
            df=df.loc[valid].copy()
            df['time']=pd.Timestamp(base)+pd.to_timedelta(df['Elapsedtime_sec'],unit='s')
            background=df.loc[df['time']<start,'CH4vmr_ppmv']
            if len(background)<30:
                raise ValueError(f'Insufficient pre-release background for beam{beam}')
            baseline=float(background.median())
            baseline_mad=float((background-baseline).abs().median())
            active=df.loc[(df['time']>=start)&(df['time']<=end)].copy()
            active['minute']=active['time'].dt.floor('min')
            grouped=active.groupby('minute',sort=True)
            aggregated=grouped[used[1:]].mean()
            aggregated['raw_sample_count']=grouped.size()
            adequate=aggregated['raw_sample_count']>=8
            aggregated=aggregated.loc[adequate].reset_index().rename(columns={'minute':'time'})
            aggregated['beam']=beam
            aggregated['baseline_ppm']=baseline
            aggregated['residual_ppm']=aggregated['CH4vmr_ppmv']-baseline
            observations.append(aggregated)
            wind_frames.append(active)
            origin=np.asarray(instruments['line_of_sight_sensor'],dtype=float)
            endpoint=np.asarray(instruments[f'reflector_{beam}'],dtype=float)
            header_length=float(lines[lines.index('Open Path length (m)')+1])
            coordinate_length=float(np.linalg.norm(endpoint-origin))
            if abs(coordinate_length-header_length)>max(0.5,0.01*header_length):
                raise ValueError(f'Beam{beam} geometry/header path lengths disagree')
            geometry.append({'beam':beam,'start_x_m':origin[0],'start_y_m':origin[1],
                'start_z_m':origin[2],'end_x_m':endpoint[0],'end_y_m':endpoint[1],'end_z_m':endpoint[2],
                'length_geometry_m':coordinate_length,'length_header_m':header_length})
            raw_records.append({'beam':beam,'archive_member':name,'sha256':hashlib.sha256(raw).hexdigest(),
                'raw_rows':len(data),'invalid_required_rows':int((~valid).sum()),
                'pre_release_rows':len(background),'baseline_ppm':baseline,'baseline_MAD_ppm':baseline_mad,
                'active_rows':len(active),'retained_minutes':len(aggregated),
                'minutes_below8_samples':int((~adequate).sum()),'original_start_label':label})
    obs=pd.concat(observations,ignore_index=True).sort_values(['time','beam'])
    obs.to_csv(output/'observations.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S')
    pd.DataFrame(geometry).to_csv(output/'beams.csv',index=False,encoding='utf-8-sig')
    wind=pd.concat(wind_frames,ignore_index=True).groupby('minute')[[
        'WindX_m/s','WindY_m/s','WindZ_m/s','Temp_K','Press_Pa']].mean().reset_index().rename(columns={'minute':'time'})
    wind.to_csv(output/'raw_axis_weather.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S')
    starts=np.asarray([[r['start_x_m'],r['start_y_m'],r['start_z_m']] for r in geometry])
    ends=np.asarray([[r['end_x_m'],r['end_y_m'],r['end_z_m']] for r in geometry])
    p64,w64=beam_quadrature(starts,ends,64)
    p128,w128=beam_quadrature(starts,ends,128)
    np.savez_compressed(output/'beam_quadrature.npz',points64=p64,weights64=w64,points128=p128,weights128=w128)
    truth=json.loads((public/'author_truth_values.json').read_text(encoding='utf-8'))
    (output/'source_truth_unmodified.json').write_text(json.dumps(truth,indent=2),encoding='utf-8')
    report={'status':'observations/geometry prepared only; no real-data inversion',
        'selection':event,
        'window_original_clock':[str(start),str(end)],'timezone':'not verified; original clock preserved, no UTC conversion',
        'averaging':'one-minute arithmetic averages; at least8 finite required samples/beam/minute',
        'background':'per-beam median of ALL available finite raw records before active window; not estimated from fit residuals',
        'negative_residuals':'retained without clipping',
        'quality':'finite required columns only; no author wind-speed>1 filter applied; not a strict processing reproduction',
        'wind':'original sonic axes retained; no from/to or coordinate conversion guessed',
        'measurements':len(obs),'beams':7,'unique_minutes':int(obs['time'].nunique()),
        'raw_records':raw_records,'raw_zip_sha256':hashlib.sha256((public/'chilbolton_original.zip').read_bytes()).hexdigest(),
        'geometry_source':'author shared primitive geometry if available; otherwise printed processing dictionary; no notebook execution',
        'height_discrepancy':metadata['height_discrepancy'],
        'still_required':['verify wind/coordinate convention','resolve source height provenance',
                          'validate temporal/path averaging with physical transport','close vertical mass-rate and ppm units',
                          'do not concatenate separated releases into one continuous PDE'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','measurements','beams','unique_minutes')},indent=2))


def verify_cylinder_mass():
    public=ROOT/'public_data'
    output=ROOT/'controlled_release_preparation'/'source1_rel03'
    target=output/'cylinder_reference.json'
    if target.exists():
        raise FileExistsError('Cylinder reference already exists; preserve it')
    name='20170509/weight data/20170509 rel03 and rel04.CSV'
    with zipfile.ZipFile(public/'chilbolton_original.zip') as archive:
        raw=archive.read(name)
    rows=[]
    for row in csv.reader(raw.decode('utf-8-sig').splitlines()):
        if not row:
            continue
        if len(row)!=3:
            raise ValueError('Unexpected cylinder mass record shape')
        timestamp=datetime.strptime(row[0].removeprefix('SHELL')+' '+row[1],'%d/%m/%y %H:%M:%S')
        rows.append({'time':timestamp,'mass_kg':float(row[2])})
    table=pd.DataFrame(rows)
    if not np.isfinite(table['mass_kg']).all() or not table['time'].is_monotonic_increasing:
        raise ValueError('Cylinder mass records are nonfinite or not time ordered')
    background=table.loc[table['time']<=pd.Timestamp('2017-05-09 12:49:00')]
    active=table.loc[(table['time']>=pd.Timestamp('2017-05-09 12:49:00'))&
                     (table['time']<=pd.Timestamp('2017-05-09 13:49:59.999999'))]
    if len(background)<5 or len(active)<10 or np.ptp(background['mass_kg'])>0.05:
        raise ValueError('Pre-window stable-mass or active-window coverage check failed')
    elapsed=(active['time']-active['time'].iloc[0]).dt.total_seconds().to_numpy()
    slope,intercept=np.polyfit(elapsed,active['mass_kg'].to_numpy(),1)
    truth=json.loads((public/'author_truth_values.json').read_text())['values_as_stored']
    table.to_csv(output/'cylinder_mass.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S')
    record={'archive_member':name,'sha256':hashlib.sha256(raw).hexdigest(),
        'date_formats':'cylinder DD/MM/YY; concentration header MM/DD/YYYY; both resolve to2017-05-09',
        'mass_unit':'kg, declared in original dataset README',
        'pre_window_mass_records':len(background),'pre_window_mass_range_kg':float(np.ptp(background['mass_kg'])),
        'pre_window_support':'recorded cylinder mass is constant before selected active window; supports pre-release background, limited by scale resolution',
        'active_records':len(active),'mass_fit_intercept_kg':float(intercept),
        'ols_emission_reference_kg_s':float(-slope),'ols_emission_reference_kg_h':float(-slope*3600),
        'author_stored_emission_value':truth['source_1_emission_rate'],
        'author_emission_unit_verification':'author numerical value agrees in scale with independently derived kg/s; not a model calibration',
        'usage':'evaluation reference only, NEVER an input amplitude calibration or parameter initialization',
        'uncertainty':'No confidence interval inferred from quantized correlated scale samples'}
    target.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


def prepare_documented_events():
    catalog=json.loads((ROOT/'public_data/author_release_windows.json').read_text(encoding='utf-8'))
    destination=ROOT/'controlled_release_preparation_v2'
    destination.mkdir(exist_ok=True)
    records=[]
    for event in catalog['events']:
        label=event['source'].replace('_','')+'_'+event['release']
        output=destination/label
        if not event['eligible']:
            records.append(dict(event=event,status='excluded_before_fitting',reason=event['exclusion']))
            continue
        stopped=output/'PREPARATION_STOPPED.json'
        if stopped.exists():
            records.append(json.loads(stopped.read_text(encoding='utf-8')))
            continue
        try:
            if not (output/'manifest.json').exists():
                main(event,output)
            elif json.loads((output/'manifest.json').read_text(encoding='utf-8'))['selection']!=event:
                raise ValueError('Prepared event metadata changed; use a new version')
        except ValueError as exc:
            # Prespecified preprocessing requirement, not a failed inversion.
            if not str(exc).startswith('Insufficient pre-release background'):
                raise
            report=dict(event=event,status='unsupported_background',reason=str(exc),
                consequence='no fabricated baseline; no inversion for this event until background is supported')
            (output/'PREPARATION_STOPPED.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            records.append(report)
            continue
        obs=pd.read_csv(output/'observations.csv')
        expected=pd.date_range(event['date']+' '+event['start'],event['date']+' '+event['end'],freq='min')
        counts=obs.groupby('time').beam.nunique()
        if len(counts)!=len(expected) or not (counts==7).all():
            raise ValueError(f'Incomplete seven-beam minute coverage: {label}; do not silently drop beams')
        records.append(dict(event=event,status='prepared',measurements=len(obs),minutes=len(counts)))
    report=dict(status='prepared observations only; no real-data inversion',events=records,
        inclusion='author-documented single-source events with >=30 pre-window records per beam and complete seven-beam minutes',
        source_selection_or_withholding=False,
        prepared_events=sum(r['status']=='prepared' for r in records))
    (destination/'catalog.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def verify_documented_cylinders():
    directory=ROOT/'controlled_release_preparation_v2'
    catalog=json.loads((directory/'catalog.json').read_text(encoding='utf-8'))
    truth=json.loads((ROOT/'public_data/author_truth_values.json').read_text(encoding='utf-8'))['values_as_stored']
    rows=[]
    with zipfile.ZipFile(ROOT/'public_data/chilbolton_original.zip') as archive:
        for case in catalog['events']:
            if case['status']!='prepared':
                continue
            event=case['event']
            name=(f'{event["date"]}/weight data/{event["date"]} rel03 and rel04.CSV'
                  if event['date']=='20170509' else f'{event["date"]}/weight data/{event["date"]}.CSV')
            raw=archive.read(name)
            values=[]
            for line in csv.reader(raw.decode('utf-8-sig').splitlines()):
                if not line:
                    continue
                if len(line)!=3:
                    raise ValueError('Unexpected cylinder row format')
                timestamp=datetime.strptime(line[0].removeprefix('SHELL')+' '+line[1],'%d/%m/%y %H:%M:%S')
                values.append(dict(time=timestamp,mass_kg=float(line[2])))
            frame=pd.DataFrame(values)
            if not np.isfinite(frame.mass_kg).all() or not frame.time.is_monotonic_increasing:
                raise ValueError('Invalid cylinder mass/time sequence')
            day=datetime.strptime(event['date'],'%Y%m%d').date().isoformat()
            start=pd.Timestamp(day+' '+event['start'])
            end=pd.Timestamp(day+' '+event['end'])+pd.Timedelta(microseconds=999999)
            active=frame[(frame.time>=start)&(frame.time<=end)]
            pre=frame[(frame.time>=start-pd.Timedelta(minutes=10))&(frame.time<start)]
            slope=intercept=None
            if len(active)>=10:
                elapsed=(active.time-active.time.iloc[0]).dt.total_seconds().to_numpy()
                slope,intercept=np.polyfit(elapsed,active.mass_kg.to_numpy(),1)
                if not -slope>0:
                    raise ValueError('Cylinder mass does not establish positive release in the selected window')
            record=dict(source=event['source'],release=event['release'],archive_member=name,
                sha256=hashlib.sha256(raw).hexdigest(),active_records=len(active),
                pre10min_records=len(pre),pre10min_range_kg=float(np.ptp(pre.mass_kg)) if len(pre) else None,
                ols_emission_reference_kg_s=None if slope is None else float(-slope),
                ols_emission_reference_kg_h=None if slope is None else float(-3600*slope),
                mass_fit_intercept_kg=None if intercept is None else float(intercept),
                event_rate_reference_status='measured' if slope is not None else 'unavailable_fewer_than10_scale_records',
                author_aggregate_rate_kg_s=truth[event['source']+'_emission_rate'],
                use='evaluation-only time-averaged reference, never used to rescale predictions or initialize fitted Q',
                uncertainty='quantized correlated scale readings; no unverified confidence interval or derivative Q(t) truth')
            target=directory/(event['source'].replace('_','')+'_'+event['release'])
            if (target/'cylinder_reference_v2.json').exists():
                if json.loads((target/'cylinder_reference_v2.json').read_text(encoding='utf-8'))!=record:
                    raise ValueError('Existing cylinder reference changed')
            else:
                if not (target/'active_cylinder_mass.csv').exists():
                    active.to_csv(target/'active_cylinder_mass.csv',index=False,date_format='%Y-%m-%d %H:%M:%S')
                (target/'cylinder_reference_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            rows.append(record)
    pd.DataFrame(rows).to_csv(directory/'cylinder_references.csv',index=False)
    print(pd.DataFrame(rows)[['source','release','active_records','ols_emission_reference_kg_h']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-cylinder',action='store_true')
    parser.add_argument('--all-events',action='store_true')
    parser.add_argument('--all-cylinder-references',action='store_true')
    args=parser.parse_args()
    if args.all_cylinder_references:
        verify_documented_cylinders()
    elif args.all_events:
        prepare_documented_events()
    elif args.verify_cylinder:
        verify_cylinder_mass()
    else:
        main()
