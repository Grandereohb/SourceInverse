"""Fetch a pinned author sample and audit inputs; never fits an inversion model."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/paper_experiments_20261017/metec_dlq_public_audit_20261027'
COMMIT = 'd289f32264d2d32045a1d486670af9a6dd136c80'
FILES = ['README.md', 'LICENSE', 'input_data/ADED_data_clean_SAMPLE.csv',
         'input_data/ADED_data_clean_README.txt', 'input_data/sensor_locations.csv',
         'input_data/source_locations.csv', 'input_data/leak_data.csv',
         'input_data/leak_data_README.xlsx', 'input_data/simulation_config.txt',
         'input_data/simulation_config_README.txt', 'input_data/DLQ_config.txt',
         'input_data/DLQ_config_README.txt', 'code/MAIN_3_analyze_results.R',
         'code/HELPER_distance_conversions.R', 'code/MAIN_1_simulate.R',
         'code/HELPER_gpuff_function.R']


def fetch():
    entries = []
    for name in FILES:
        url = f'https://raw.githubusercontent.com/wsdaniels/DLQ/{COMMIT}/{name}'
        path = OUT/'raw'/name
        if not path.exists():
            with urlopen(Request(url, headers={'User-Agent': 'SourceInverse-data-audit'}), timeout=60) as response:
                content = response.read(64*1024*1024+1)
            if len(content) > 64*1024*1024:
                raise ValueError('Unexpectedly large single input; inspect before continuing')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        entries.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), url=url))
        print('FETCHED', name, len(content), flush=True)
    (OUT/'provenance.json').write_text(json.dumps(dict(repository='https://github.com/wsdaniels/DLQ',
        commit=COMMIT, client_audit_date='2026-10-27', files=entries,
        scope='public sample only; not full proprietary concentration dataset'), indent=2), encoding='utf-8')


def inspect():
    for name in ['ADED_data_clean_SAMPLE.csv', 'sensor_locations.csv', 'source_locations.csv', 'leak_data.csv']:
        path = OUT/'raw/input_data'/name
        frame = pd.read_csv(path)
        print('\nFILE', name, 'SHAPE', frame.shape, 'COLUMNS', frame.columns.tolist())
        print(frame.head(3).to_string(index=False))
        print('MISSING', frame.isna().sum().to_dict())
    for name in ['ADED_data_clean_README.txt', 'simulation_config.txt', 'DLQ_config.txt']:
        print('\nTEXT', name)
        print((OUT/'raw/input_data'/name).read_text(encoding='utf-8')[:10000])
    book = pd.ExcelFile(OUT/'raw/input_data/leak_data_README.xlsx')
    for sheet in book.sheet_names:
        print('\nTRUTH DICTIONARY', sheet)
        print(pd.read_excel(book, sheet_name=sheet).to_string(index=False))


def longest_missing(values):
    longest = current = 0
    for available in values:
        current = 0 if available else current+1
        longest = max(longest, current)
    return longest


def audit():
    raw = OUT/'raw/input_data'
    observations = pd.read_csv(raw/'ADED_data_clean_SAMPLE.csv')
    observations['timestamp'] = pd.to_datetime(observations.time, utc=True)
    observations['wind_valid'] = np.isfinite(observations['wind.speed']) & np.isfinite(observations['wind.direction'])
    sensors = pd.read_csv(raw/'sensor_locations.csv')
    sources = pd.read_csv(raw/'source_locations.csv')
    leaks = pd.read_csv(raw/'leak_data.csv')
    leaks['start'] = pd.to_datetime(leaks.tc_ExpStartDatetime, utc=True)
    leaks['end'] = pd.to_datetime(leaks.tc_ExpEndDatetime, utc=True)
    clock = pd.date_range(observations.timestamp.min(), observations.timestamp.max(), freq='min')
    station_count = observations.groupby('timestamp').name.nunique().reindex(clock, fill_value=0)
    wind_valid = observations.groupby('timestamp').wind_valid.any().reindex(clock, fill_value=False)
    duplicates = observations[observations.duplicated(['timestamp', 'name'], keep=False)]
    duplicates.to_csv(OUT/'duplicate_records.csv', index=False)
    station_rows = []
    for name, sample in observations.groupby('name'):
        reference = sensors.set_index('name').loc[name]
        delta = sample[['lon','lat']].to_numpy()-reference[['lon','lat']].to_numpy(dtype=float)
        delta_m = np.deg2rad(delta)*6371000*np.array([np.cos(np.deg2rad(reference.lat)),1.])
        station_rows.append(dict(station=name, rows=len(sample), unique_minutes=sample.timestamp.nunique(),
            absent_minutes=len(clock)-sample.timestamp.nunique(), wind_valid_rows=int(sample.wind_valid.sum()),
            sensor_coordinate_matches=bool(np.allclose(sample[['lon','lat']].to_numpy(),
                sensors.set_index('name').loc[name,['lon','lat']].to_numpy(dtype=float), rtol=0., atol=1e-9)),
            coordinate_max_difference_m=float(np.linalg.norm(delta_m,axis=1).max()),
            concentration_missing=int(sample.methane.isna().sum()), height_m=float(reference.height)))
    pd.DataFrame(station_rows).to_csv(OUT/'station_coverage.csv', index=False)
    overlapping = leaks[(leaks.start < clock[-1]) & (leaks.end > clock[0])].sort_values('start')
    # Preserve the author's explicit group-ID mapping, not nearest-source classification.
    mapping = dict(zip(leaks.tc_EquipmentGroupID.unique(),
        ['Tank','Wellhead.East','Wellhead.West','Separator.East','Separator.West']))
    events = []
    for _, release in overlapping.iterrows():
        stamps = pd.date_range(release.start.ceil('min'), release.end.floor('min'), freq='min')
        available = station_count.reindex(stamps, fill_value=0)
        wind = wind_valid.reindex(stamps, fill_value=False)
        concurrent = overlapping[(overlapping.start < release.end) & (overlapping.end > release.start)]
        group = mapping[release.tc_EquipmentGroupID]
        center = sources.set_index('name').loc[group]
        lat_mean = np.deg2rad((center.lat+release.tc_Latitude)/2)
        offset = np.array([np.deg2rad(center.lon-release.tc_Longitude)*np.cos(lat_mean),
                           np.deg2rad(center.lat-release.tc_Latitude)])*6371000
        duplicate_in_event = duplicates[duplicates.timestamp.isin(stamps)]
        events.append(dict(experiment_id=int(release.tc_ExperimentID), emission_point=release.tc_EPID,
            equipment_group=group, start_utc=release.start.isoformat(), end_utc=release.end.isoformat(),
            wholly_inside_sample=bool(release.start >= clock[0] and release.end <= clock[-1]),
            emission_points_declared=int(release.tc_ExpEPCount), overlapping_release_rows=len(concurrent),
            latitude=float(release.tc_Latitude), longitude=float(release.tc_Longitude),
            group_reference_height_m=float(center.height), height_is_group_reference_not_individual_survey=True,
            group_center_offset_m=float(np.linalg.norm(offset)),
            mean_rate_kg_h=float(release.tc_C1MassFlow/1000),
            rate_uncertainty_kg_h=float(release.tc_C1MassFlowUncertainty/1000),
            time_stamps=len(stamps), complete8_minutes=int((available == 8).sum()),
            min_sensor_count=int(available.min()), no_wind_minutes=int((~wind).sum()),
            longest_missing_wind_minutes=longest_missing(wind),
            duplicate_extra_rows=len(duplicate_in_event)-len(duplicate_in_event.drop_duplicates(['timestamp','name'])),
            full8_with_wind=bool((available == 8).all() and wind.all()),
            at_least6_with_wind=bool((available >= 6).all() and wind.all())))
    events = pd.DataFrame(events)
    events.to_csv(OUT/'release_event_audit.csv', index=False)
    dictionary = pd.read_excel(raw/'leak_data_README.xlsx')
    needed = dictionary[dictionary.Header.astype(str).str.replace(':','',regex=False).str.strip().isin(
        ['tc_C1MassFlow','tc_C1MassFlowUncertainty','tc_Latitude','tc_Longitude','tc_ExpStartDatetime','tc_ExpEndDatetime'])]
    needed.to_csv(OUT/'truth_field_dictionary.csv', index=False)
    summary = dict(audit_complete=True, inversion_experiments_run=0, pinned_commit=COMMIT,
        observation_file_bytes=(raw/'ADED_data_clean_SAMPLE.csv').stat().st_size,
        total_download_bytes=sum(p.stat().st_size for p in (OUT/'raw').rglob('*') if p.is_file()),
        observation_rows=len(observations), sensor_count=len(sensors), wind_stations=['E','NW','SW'],
        utc_start=clock[0].isoformat(), utc_end=clock[-1].isoformat(),
        minute_grid_size=len(clock), observed_distinct_minutes=observations.timestamp.nunique(),
        duplicate_extra_rows=len(observations)-len(observations.drop_duplicates(['timestamp','name'])),
        missing_sensor_minute_pairs=len(clock)*len(sensors)-len(observations.drop_duplicates(['timestamp','name'])),
        complete8_minutes=int((station_count == 8).sum()), at_least6_minutes=int((station_count >= 6).sum()),
        all_sensor_absent_minutes=int((station_count == 0).sum()), any_wind_minutes=int(wind_valid.sum()),
        no_wind_minutes=int((~wind_valid).sum()),
        reference_sensor_coordinates_exact_match=all(x['sensor_coordinate_matches'] for x in station_rows),
        max_sensor_coordinate_difference_m=max(x['coordinate_max_difference_m'] for x in station_rows),
        truth_file_rows=len(leaks), overlapping_release_rows=len(events),
        overlapping_experiments=int(events.experiment_id.nunique()),
        wholly_inside_sample=int(events.wholly_inside_sample.sum()),
        events_single_point_and_no_overlap=int(((events.emission_points_declared == 1)&(events.overlapping_release_rows == 1)).sum()),
        equipment_groups=int(events.equipment_group.nunique()),
        distinct_recorded_release_coordinates=int(events[['latitude','longitude']].drop_duplicates().shape[0]),
        release_rate_kg_h_range=[float(events.mean_rate_kg_h.min()),float(events.mean_rate_kg_h.max())],
        full8_no_wind_gap_events=int(events.full8_with_wind.sum()),
        at_least6_no_wind_gap_events=int(events.at_least6_with_wind.sum()),
        strict8_event_ids=events.loc[events.full8_with_wind,'experiment_id'].tolist(),
        group_center_offset_m_range=[float(events.group_center_offset_m.min()),float(events.group_center_offset_m.max())],
        conclusion='Conditionally usable for independent point-sensor source localization after fixed QC and a height-aware physical adapter; not ready-to-run production input',
        unresolved=['Duplicate rows require a documented aggregation rule; no row silently removed by this audit',
            'A missing station is not a zero concentration; no wind gap silently interpolated',
            'Raw methane includes background; background rule needed before inversion',
            'Public sample has no synchronous temperature/pressure; author uses fixed methane ppm factor1.524e6',
            'Sensor heights2.4m and group source heights2/4.5m require vertical diffusion and ground reflection',
            'Use per-release WGS84 truth, not equipment-group center for metre error; coordinate survey uncertainty unspecified',
            'Author source localization is discrete equipment-group classification; cannot compare its published percentage to our metre errors',
            'Public sample only; full-month concentration requires author request; MIT file explicitly licenses software, no separate dataset licence found'])
    (OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(events[['experiment_id','equipment_group','time_stamps','min_sensor_count','no_wind_minutes',
                  'duplicate_extra_rows','full8_with_wind','at_least6_with_wind']].to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    fetch()
    if args.inspect:
        inspect()
    audit()
