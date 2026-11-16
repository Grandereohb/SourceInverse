"""Run one inversion per merged alarm event, without regrouping by pollutant."""
from pathlib import Path
import argparse
import tempfile
import sys
import pandas as pd
import extract_monitor_data_shsh_js as extract
import run_recent_leak_source_inversions as batch

ROOT = Path(__file__).resolve().parents[1]
# Manual inputs. Generate ALARM_FILE with merge_station_alarm_events_zhenhua.py first.
ALARM_FILE = ROOT / 'data/zhenhua1112/merged_alarm_events_q1_by_pollutant.xlsx'
MONITOR_FILE = ROOT / 'data/zhenhua1112/振华前三季度数据/振华第一季度数据/2026年第一季度空气站历史数据.xlsx'
SITES_FILE = ROOT / 'data/zhenhua1112/sites.xlsx'
SOURCE_INVERSION_COUNT = 10
TRAVERSE_DIRECTION = 'backward'
START_TRAVERSE_TIME = ''
MANUAL_INPUT = []
POLLUTANT_CONTAINS = ''
PAD_HOURS = 6
CONTINUE_ON_ERROR = True
MONITOR_CONCENTRATION_UNIT = 'ppb'  # History workbook; alarm values use ug/m3, not fit targets.
# Explicit sums only: all component columns must exist; never replace missing data by 0.
POLLUTANT_COMPONENTS = {'二甲苯': ['邻-二甲苯', '间-二甲苯+对-二甲苯']}
STATION_ALIASES = {'1号生活区站': '振华1号生活区站', '科技楼站': '振华科技楼站',
                   '2000亩中心站': '振华2000亩中心站'}


def alarm_events(path, pad_hours=PAD_HOURS):
    if pad_hours < 0:
        raise ValueError('pad_hours must be nonnegative')
    frame = pd.read_excel(path, sheet_name='abnormal_high_records')
    required = {'station', 'pollutant', 'time', 'concentration', 'event_start', 'event_end'}
    if missing := required-set(frame.columns):
        raise ValueError(f'Not a merged alarm workbook: missing {sorted(missing)}')
    events = []
    for _, row in frame.iterrows():
        start, end, time = [pd.Timestamp(row[k]) for k in ['event_start', 'event_end', 'time']]
        if any(pd.isna(t) for t in [start, end, time]) or not start <= time <= end:
            raise ValueError('Invalid event timestamps')
        events.append(dict(pollutant=str(row.pollutant), wind_station=str(row.station),
            leak_start=start, leak_end=end, representative_time=time,
            start_time=start-pd.Timedelta(hours=pad_hours), end_time=end+pd.Timedelta(hours=pad_hours),
            max_concentration=float(row.concentration),
            n_hours=int((end-start).total_seconds()/3600)+1, n_records=int(row.get('alarm_count', 1))))
    return events


def concentration_table(workbook, event):
    components = POLLUTANT_COMPONENTS.get(event['pollutant'], [event['pollutant']])
    tables = [extract.build_concentration_table(workbook, event['start_time'], event['end_time'], name)
              .set_index('时间').drop(columns='TARGET_POLLUTANT') for name in components]
    stations = [c for c in tables[0] if all(c in t for t in tables)]
    if not stations:
        raise ValueError('No stations with all pollutant component columns')
    result = tables[0][stations].copy()
    for table in tables[1:]:
        result = result + table[stations]
    result = result.dropna(axis=1, how='all')
    if event['wind_station'] not in result.columns:
        raise ValueError('Alarm station has no usable target concentration')
    result['TARGET_POLLUTANT'] = event['pollutant']
    return result.reset_index()


def run(dry_run=False, count=SOURCE_INVERSION_COUNT):
    selected = batch.select_leaks(alarm_events(ALARM_FILE), count, TRAVERSE_DIRECTION,
        start_time=START_TRAVERSE_TIME, manual_input=MANUAL_INPUT, pollutant_contains=POLLUTANT_CONTAINS)
    if not selected:
        raise ValueError('No alarm events matched filters')
    for rank, event in selected:
        print(rank, event['pollutant'], event['wind_station'], event['representative_time'],
              event['start_time'], event['end_time'])
    if dry_run:
        return
    # Reuse the model's loader, including lon/lat rows and degree-minute-second coordinates.
    sys.path.insert(0, str(ROOT/'pinn_source'))
    from data_io import load_sites
    sites, _, _ = load_sites(SITES_FILE)
    sites['station'] = sites['station'].map(extract.clean_station_name)
    sites['station'] = sites['station'].replace(STATION_ALIASES)
    if sites.station.duplicated().any():
        raise ValueError('Duplicate station names in sites workbook')
    ROOT.joinpath('result').mkdir(exist_ok=True)
    summary_path = ROOT / f'result/alarm_run_summary_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx'
    config_before = batch.PINN_CONFIG.read_bytes()
    rows = []
    try:
        with pd.ExcelFile(MONITOR_FILE) as workbook, tempfile.TemporaryDirectory(prefix='alarm_inputs_') as temp:
            for rank, event in selected:
                folder = Path(temp)/f'event_{rank:04d}'
                folder.mkdir()
                record = dict(event_rank=rank, **event, status='failed',
                    monitor_file=str(MONITOR_FILE), sites_file=str(SITES_FILE),
                    monitor_concentration_unit=MONITOR_CONCENTRATION_UNIT)
                try:
                    conc = concentration_table(workbook, event)
                    wind = extract.build_wind_table(workbook, event['start_time'], event['end_time'], event['wind_station'])
                    station_names = [c for c in conc if c not in {'时间', 'TARGET_POLLUTANT'}]
                    record['n_stations'] = len(station_names)
                    if len(station_names) < 2:
                        print('WARNING: only one usable concentration station; location may be weakly constrained.', flush=True)
                    missing = set(station_names)-set(sites.station)
                    if missing:
                        raise ValueError(f'Station coordinates missing: {sorted(missing)}')
                    if conc.empty or wind.empty:
                        raise ValueError('Empty concentration or wind window')
                    paths = {key: folder/name for key, name in [('SITE_PATH', 'sites.xlsx'),
                        ('CONC_PATH', 'concentration.xlsx'), ('WIND_PATH', 'wind.xlsx')]}
                    sites.set_index('station').loc[station_names].reset_index().to_excel(paths['SITE_PATH'], index=False)
                    conc.to_excel(paths['CONC_PATH'], index=False)
                    wind.to_excel(paths['WIND_PATH'], index=False)
                    batch.update_pinn_config_inputs(event, extracted_paths=paths)
                    before = {p for p in batch.RESULT_DIR.iterdir() if p.is_dir()}
                    batch.run_step([sys.executable, str(batch.PINN_SCRIPT)], folder/'pinn_source_pinn.log')
                    result = batch.latest_result_dir(before)
                    (folder/'pinn_source_pinn.log').replace(result/'pinn_source_pinn.log')
                    record.update(status='completed', result_dir=str(result))
                except Exception as exc:
                    record['error'] = str(exc)
                    # Keep failed inputs and logs outside the temporary folder.
                    import shutil
                    failure = summary_path.parent/(summary_path.stem+'_failed')/f'event_{rank:04d}'
                    shutil.copytree(folder, failure, dirs_exist_ok=True)
                    record['failure_dir'] = str(failure)
                    print('FAILED', rank, str(exc), flush=True)
                    if not CONTINUE_ON_ERROR:
                        raise
                finally:
                    rows.append(record)
                    pd.DataFrame(rows).to_excel(summary_path, index=False)
    finally:
        batch.PINN_CONFIG.write_bytes(config_before)
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='List events without extraction or training')
    parser.add_argument('--count', type=int, default=SOURCE_INVERSION_COUNT)
    args = parser.parse_args()
    run(args.dry_run, args.count)
