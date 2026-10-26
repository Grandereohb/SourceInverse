"""Read-only industrial application cohort preparation; no source truth or fits.

Catalog selection precedes inversion. All eligible episodes are retained, with
shared-time clusters recorded rather than calling correlated events independent.
Unknown instrument codes are preserved, never assigned an invented meaning.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import extract_monitor_data_jjj as extractor

DESTINATION = ROOT/'result/paper_experiments_20261017/industrial_preparation_v2_flags'
MONITOR = ROOT/'data/jjj/2026年小时数据/567月小时数据_标准单位_汇总.xlsx'
CATALOG = ROOT/'data/abnormal_high_monitor_data/abnormal_high_monitor_data_jjj_567.xlsx'
SITES = ROOT/'data/jjj/2026年小时数据/当前数据点位信息.xlsx'
TIME = '时间'
PATTERN = re.compile(r'^\s*(<=|>=|<|>|≤|≥)?\s*'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*'
    r'(?:[（(]([^()（）]+)[)）])?\s*$')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_measurement(value):
    """Only full numeric literals with optional relation/status are accepted."""
    if pd.isna(value):
        return dict(raw_text=None, reported_value=None, point_value=None,
                    relation=None, status=None, state='missing')
    text = str(value).strip().replace('−', '-')
    match = PATTERN.fullmatch(text)
    if match is None:
        return dict(raw_text=str(value), reported_value=None, point_value=None,
                    relation=None, status=None,
                    state='missing' if text in ('', '--', '---') else 'unparsed')
    relation, number, status = match.groups()
    number = float(number)
    finite = bool(np.isfinite(number))
    return dict(raw_text=str(value), reported_value=number if finite else None,
        point_value=number if finite and relation is None else None,
        relation=relation, status=status,
        state=('nonfinite' if not finite else 'bounded' if relation else 'point_numeric'))


def merge_catalog_events(records, gap_hours=12, padding_hours=6):
    required = {'station', 'pollutant', 'time', 'concentration'}
    if not required.issubset(records.columns):
        raise ValueError('Catalog lacks required event fields')
    records = records.copy()
    records['time'] = pd.to_datetime(records.time, errors='raise')
    records['concentration'] = pd.to_numeric(records.concentration, errors='raise')
    if (records.time.isna().any() or not np.isfinite(records.concentration).all()
            or records.station.isna().any() or records.pollutant.isna().any()):
        raise ValueError('Invalid catalog rows cannot be silently removed')
    merged = []
    for pollutant, group in records.groupby('pollutant', sort=True):
        group = group.sort_values(['time', 'station'], kind='stable')
        block = (group.time.diff()>pd.Timedelta(hours=gap_hours)).cumsum()
        for _, event in group.groupby(block, sort=True):
            peak = event.sort_values(['concentration', 'time', 'station'],
                                    ascending=[False, True, True], kind='stable').iloc[0]
            first, last = event.time.min(), event.time.max()
            merged.append(dict(pollutant=str(pollutant), trigger_start=first,
                trigger_end=last, window_start=first-pd.Timedelta(hours=padding_hours),
                window_end=last+pd.Timedelta(hours=padding_hours),
                peak_time=peak.time, peak_station=str(peak.station),
                catalog_peak=float(peak.concentration), catalog_row_count=len(event),
                catalog_rows=event.to_dict('records')))
    merged.sort(key=lambda v:(v['trigger_start'], v['pollutant']))
    for index, event in enumerate(merged):
        event['case_id'] = f'ind_{index+1:03d}'
    return merged


def temporal_clusters(cases, baseline_hours=6):
    """Transitive overlap of baseline+fit supports; not an independence proof."""
    clusters = {}
    end = None
    cluster = 0
    for event in sorted(cases, key=lambda v:(v['window_start'], v['case_id'])):
        start = pd.Timestamp(event['window_start'])-pd.Timedelta(hours=baseline_hours)
        stop = pd.Timestamp(event['window_end'])
        if end is None or start>end:
            cluster += 1
            end = stop
        else:
            end = max(end, stop)
        clusters[event['case_id']] = f'episode_{cluster:03d}'
    return clusters


def read_monitor(path, pollutants):
    records, inventory, tables, units = [], [], {}, {}
    with pd.ExcelFile(path) as workbook:
        for sheet in workbook.sheet_names:
            station = extractor.clean_station_name(sheet)
            if station in tables:
                raise ValueError('Duplicate cleaned station name: '+station)
            table = pd.read_excel(workbook, sheet_name=sheet)
            table.columns = [str(v).strip() for v in table.columns]
            if TIME not in table:
                raise ValueError('Missing clock in '+sheet)
            unit_rows = table[table[TIME].astype(str).str.strip()=='标准单位']
            station_units = unit_rows.iloc[0].to_dict() if len(unit_rows)==1 else {}
            table['parsed_time'] = pd.to_datetime(table[TIME], errors='coerce', format='mixed')
            invalid_time = table.parsed_time.isna()
            non_unit_bad = int((invalid_time & table[TIME].notna() &
                table[TIME].astype(str).str.strip().ne('标准单位')).sum())
            if non_unit_bad:
                raise ValueError('Unrecognized non-unit timestamp rows in '+sheet)
            table['excel_row'] = table.index+2
            table = table[~invalid_time].copy()
            if not table.parsed_time.eq(table.parsed_time.dt.floor('h')).all():
                raise ValueError('Non-hourly labels require explicit handling: '+sheet)
            columns = [v for v in table.columns if v in pollutants or v in ('风速', '风向')]
            values = pd.DataFrame(index=table.parsed_time)
            for column in columns:
                parsed = [parse_measurement(v) for v in table[column]]
                values[column] = [v['point_value'] for v in parsed]
                unit = station_units.get(column)
                units[station, column] = None if pd.isna(unit) else str(unit).strip()
                for row, value in zip(table.itertuples(index=False), parsed):
                    # Provenance uses the original Excel row before clock filtering.
                    records.append(dict(station=station, variable=column,
                        time=str(row.parsed_time), source_sheet=sheet,
                        source_excel_row=int(row.excel_row), unit=units[station, column], **value))
            tables[station] = values.groupby(level=0).mean().sort_index()
            inventory.append(dict(station=station, sheet=sheet, timestamp_rows=len(table),
                distinct_timestamps=table.parsed_time.nunique(), duplicate_rows=int(table.parsed_time.duplicated().sum()),
                first_time=str(table.parsed_time.min()), last_time=str(table.parsed_time.max()),
                pollutant_count=sum(c in pollutants for c in columns)))
            print('READ_STATION', station, len(table), flush=True)
    return tables, units, pd.DataFrame(records), pd.DataFrame(inventory)


def prepare_case(event, tables, units, coords, raw_records):
    labels = pd.date_range(event['window_start'], event['window_end'], freq='h')
    baseline_labels = pd.date_range(labels[0]-pd.Timedelta(hours=6), periods=6, freq='h')
    reasons = []
    if len(labels)>48:
        reasons.append('event_window_longer_than_48_hours')
    if any(name not in coords for name in tables):
        reasons.append('monitor_station_without_coordinate')
    pollutant = event['pollutant']
    station_names = [s for s, t in tables.items() if pollutant in t and
                     t[pollutant].reindex(labels).notna().any()]
    if len(station_names)<3:
        reasons.append('fewer_than_three_stations_with_point_data')
    accepted_units = {'μg/m³', 'µg/m³', 'ug/m3', 'μg/m3', 'µg/m3'}
    if any(units.get((s,pollutant)) not in accepted_units for s in station_names):
        reasons.append('unconfirmed_concentration_unit')
    wind_station = event['peak_station']
    wind_table = tables.get(wind_station, pd.DataFrame())
    if not {'风速', '风向'}.issubset(wind_table.columns):
        reasons.append('peak_station_wind_missing')
        sp, direction = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    else:
        sp = wind_table['风速'].reindex(labels).to_numpy(dtype=float)
        direction = wind_table['风向'].reindex(labels).to_numpy(dtype=float)
        if not (np.isfinite(sp).all() and np.isfinite(direction).all() and
                np.all(sp>=0) and np.all((direction>=0)&(direction<=360))):
            reasons.append('incomplete_or_invalid_hourly_wind_no_interpolation')
        if units.get((wind_station,'风速'))!='m/s' or units.get((wind_station,'风向'))!='°':
            reasons.append('unconfirmed_wind_units')
    matrix = np.column_stack([tables[s][pollutant].reindex(labels).to_numpy(dtype=float)
                              for s in station_names]) if station_names else np.empty((len(labels),0))
    before = np.column_stack([tables[s][pollutant].reindex(baseline_labels).to_numpy(dtype=float)
                              for s in station_names]) if station_names else np.empty((6,0))
    if len(station_names) and not np.isfinite(before).all():
        reasons.append('six_pre_window_background_samples_missing')
    window_raw = raw_records[(raw_records.variable==pollutant) &
        (raw_records.time>=str(baseline_labels[0])) & (raw_records.time<=str(labels[-1]))]
    unconfirmed = 0
    for item in event['catalog_rows']:
        matches = window_raw[(window_raw.station==str(item['station'])) &
                             (window_raw.time==str(item['time']))]
        if not any(pd.notna(v) and np.isclose(v,float(item['concentration']),rtol=1e-9,atol=1e-8)
                   for v in matches.point_value):
            unconfirmed += 1
    if unconfirmed:
        reasons.append('catalog_records_not_confirmed_as_raw_point_values')
    selected = window_raw[window_raw.station.isin(station_names)]
    metadata = {k:(str(v) if isinstance(v,pd.Timestamp) else v) for k,v in event.items() if k!='catalog_rows'}
    metadata.update(eligible=not reasons, rejection_reasons=reasons,
        station_names=station_names, n_stations=len(station_names), n_hours=len(labels),
        point_observation_count=int(np.isfinite(matrix).sum()),
        unconfirmed_catalog_records=unconfirmed,
        missing_point_cells=int((~np.isfinite(matrix)).sum()),
        baseline_start=str(baseline_labels[0]), baseline_end=str(baseline_labels[-1]),
        concentration_unit='ug/m3', status_code_counts=dict(Counter(selected.status.fillna('<unflagged>'))),
        value_state_counts=dict(Counter(selected.state)),
        status_interpretation='All codes unknown; numerical records are not certified instrument-valid',
        source_truth_available=False, source_rate_units='effective 2D model units, NOT kg/h')
    if reasons:
        return metadata, None
    background = np.median(before,axis=0)
    mad = np.median(np.abs(before-background),axis=0)
    residual = matrix-background
    mask = np.isfinite(matrix)
    scale = float(np.max(residual[mask]))
    if scale<=0:
        metadata.update(eligible=False,rejection_reasons=['no_positive_background_increment'])
        return metadata, None
    peaks = np.sort(np.maximum(np.nanmax(residual,axis=0),0.))
    metadata.update(concentration_scale=scale,
        peak_ratio_second_to_first=float(peaks[-2]/peaks[-1]),
        wind_speed_mean_mps=float(sp.mean()), wind_speed_min_mps=float(sp.min()),
        wind_direction_resultant=float(np.abs(np.exp(1j*np.deg2rad(direction)).mean())))
    arrays = dict(labels=labels.strftime('%Y-%m-%d %H:%M:%S').to_numpy(dtype='U19'),
        station_names=np.asarray(station_names,dtype='U'),
        longitude=np.asarray([coords[s][0] for s in station_names]),
        latitude=np.asarray([coords[s][1] for s in station_names]),
        raw_concentration=matrix, residual_concentration=residual,
        normalized_concentration=residual/scale, observation_mask=mask,
        baseline=background, baseline_mad=mad, pre_window_concentration=before,
        wind_speed_mps=sp, wind_direction_deg=direction)
    return metadata, arrays


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous preparation; use a new version directory')
    source_files = [MONITOR,CATALOG,SITES]
    files = [Path(__file__).resolve(),Path(extractor.__file__).resolve(),
             ROOT/'pinn_source/observation_quality.py']
    protocol = dict(scope='Industrial4.4 preparation only, not a fitted or blind evaluation',
        selection='All eligible merged events in the existing567 catalog; no outcome-based count truncation',
        gap_hours=12,padding_hours=6,baseline_hours=6,max_window_hours=48,
        minimum_stations=3,background='Per-station median of six complete pre-window hours',
        fit_data='Signed residuals, all finite point observations; true missing/bounds not filled with zeros',
        numeric_parser='Full scientific numeric literal; status preserved with unknown meaning; bounds retained but not point values',
        duplicate_rule='Arithmetic mean of available point records at station/variable/time; all raw records archived',
        wind='Peak-catalog station, complete co-timed hourly observations; zero speed valid, no interpolation',
        clock='Raw naive local labels preserved; concentration/wind assumed co-timed hourly means, end-label convention for future operator',
        clock_caution='Provider has not confirmed interval-label convention/timezone; absolute release times remain conditional',
        independence='Transitive overlap of baseline+fit supports defines clusters; not independent physical-source geometries',
        truth='No known source coordinates or physical release-rate truth',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in files},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_files})
    output.mkdir(parents=True)
    write_json(output/'protocol.json',protocol)
    for source in files:
        target=output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    originals=output/'source_inputs'
    originals.mkdir()
    for source,name in zip(source_files,('monitor.xlsx','catalog.xlsx','sites.xlsx')):
        shutil.copyfile(source,originals/name)
    catalog=pd.read_excel(CATALOG,sheet_name='abnormal_high_records')
    events=merge_catalog_events(catalog)
    tables,units,raw,inventory=read_monitor(MONITOR,set(catalog.pollutant))
    raw.to_csv(output/'raw_measurements.csv.gz',index=False,compression='gzip')
    inventory.to_csv(output/'station_inventory.csv',index=False)
    site_table=extractor.load_sites_table(SITES,station_order=list(tables))
    coords={s:(float(site_table[s].iloc[0]),float(site_table[s].iloc[1])) for s in site_table if s!='station'}
    if any(not (np.isfinite(lon) and np.isfinite(lat) and abs(lon)<=180 and abs(lat)<=90)
           for lon,lat in coords.values()):
        raise ValueError('Invalid station coordinates')
    results=[]
    for event in events:
        meta,arrays=prepare_case(event,tables,units,coords,raw)
        results.append(meta)
        if arrays is not None:
            case=output/meta['case_id']
            case.mkdir()
            np.savez_compressed(case/'input.npz',**arrays)
            write_json(case/'input.json',meta)
    clusters=temporal_clusters([v for v in results if v['eligible']])
    for meta in results:
        meta['temporal_cluster']=clusters.get(meta['case_id'])
        if meta['eligible']:
            write_json(output/meta['case_id']/'input.json',meta)
    pd.DataFrame(results).to_csv(output/'case_manifest.csv',index=False)
    eligible=[v for v in results if v['eligible']]
    representative=[]
    for key,reverse in (('peak_ratio_second_to_first',False),('peak_ratio_second_to_first',True),
                        ('wind_direction_resultant',False)):
        pool=[v for v in eligible if v['case_id'] not in representative]
        if pool:
            representative.append(sorted(pool,key=lambda v:((-1 if reverse else 1)*v[key],v['case_id']))[0]['case_id'])
    prepared_files=[p for v in eligible for p in
                    (output/v['case_id']/'input.json',output/v['case_id']/'input.npz')]
    report=dict(ready_for_numerical_preflight=bool(eligible),fitting_performed=False,
        catalog_rows=len(catalog),catalog_pollutants=sorted(catalog.pollutant.unique().tolist()),
        merged_candidates=len(events),eligible_cases=len(eligible),
        rejected_cases=len(events)-len(eligible),temporal_clusters=len(set(clusters.values())),
        rejection_counts=dict(Counter(reason for v in results for reason in v['rejection_reasons'])),
        representative_case_ids=representative,
        representative_rule='Lowest peak ratio, highest peak ratio, lowest wind resultant, distinct cases, no fitted outcomes',
        prepared_sha256={str(p.relative_to(output)):digest(p) for p in prepared_files},
        caveat='Eligibility is numerical/data-availability screening, not certified instrument QC or source validation')
    if any(digest(ROOT/name)!=value for name,value in protocol['input_sha256'].items()):
        raise ValueError('Source workbook changed during preparation')
    write_json(output/'PREPARATION_FINISHED.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DESTINATION)
    run(parser.parse_args().output)
