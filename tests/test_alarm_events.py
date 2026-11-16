import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from merge_station_alarm_events_zhenhua import merge_alarms, REFERENCE_COLUMNS
from run_alarm_source_inversions_zhenhua import alarm_events, concentration_table
from tempfile import TemporaryDirectory


def alarms(hours, station='A', pollutant='P'):
    return pd.DataFrame({'报警产生时间': [pd.Timestamp('2026-03-29')+pd.Timedelta(hours=h) for h in hours],
        '站点': station, '污染物': pollutant, '污染物浓度': list(range(len(hours)))})


class AlarmTests(unittest.TestCase):
    def test_threshold_inclusive_and_empty_schema(self):
        self.assertEqual(len(merge_alarms(alarms([1,2,3]), min_concentration=1)), 1)
        empty = merge_alarms(alarms([1,2,3]), min_concentration=2)
        self.assertTrue(empty.empty)
        self.assertEqual(list(empty.columns), REFERENCE_COLUMNS)

    def test_threshold_does_not_change_midpoint_or_boundaries(self):
        data = alarms([1,2,3])
        data['污染物浓度'] = [100, 1, 100]
        self.assertTrue(merge_alarms(data, min_concentration=50).empty)
        self.assertEqual(len(merge_alarms(data)), 1)

    def test_invalid_threshold(self):
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                merge_alarms(alarms([1]), min_concentration=value)

    def test_missing_hour_and_later_midpoint(self):
        result = merge_alarms(alarms([12,13,14,16,17,18,19]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].time.hour, 16)
        self.assertEqual(result.iloc[0].concentration, 3)
        self.assertEqual(list(result.columns), REFERENCE_COLUMNS)

    def test_separate_events_and_groups(self):
        data = pd.concat([alarms([1,2,6]), alarms([1],station='B'),alarms([1],pollutant='Q')])
        self.assertEqual(len(merge_alarms(data)), 3)

    def test_duplicate_times(self):
        result = merge_alarms(alarms([1,1,2]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].time.hour, 2)

    def test_batch_accepts_cross_station_merged_event(self):
        result = merge_alarms(pd.concat([alarms([12]),alarms([12],station='B')]))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.xlsx'
            result.to_excel(path,sheet_name='abnormal_high_records',index=False)
            events = alarm_events(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['start_time'].hour, 6)

    def test_cross_station_chain_and_concentration_tie(self):
        data = pd.concat([alarms([12,14,16],station='A'), alarms([13,15,17],station='B')])
        data.loc[data['站点']=='B','污染物浓度'] = 100
        row = merge_alarms(data).iloc[0]
        self.assertEqual(row.time.hour, 15)
        self.assertEqual(row.station, 'B')
        same_time = pd.concat([alarms([15],station='A'),alarms([15],station='B')])
        same_time.loc[same_time['站点']=='B','污染物浓度']=100
        self.assertEqual(merge_alarms(same_time).iloc[0].station,'B')

    def test_invalid_time(self):
        data = alarms([1])
        data.loc[0,'报警产生时间'] = pd.NaT
        with self.assertRaises(ValueError):
            merge_alarms(data)

    def test_xylene_sum_preserves_missing_component(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'monitor.xlsx'
            pd.DataFrame({'时间':pd.date_range('2026-03-29',periods=2,freq='h'),
                '邻-二甲苯':[1,2], '间-二甲苯+对-二甲苯':[3,float('nan')]}).to_excel(
                    path,sheet_name='A(带标识)',index=False)
            with pd.ExcelFile(path) as workbook:
                result=concentration_table(workbook,dict(pollutant='二甲苯',wind_station='A',
                    start_time='2026-03-29',end_time='2026-03-30'))
            self.assertEqual(result.loc[0,'A'],4)
            self.assertTrue(pd.isna(result.loc[1,'A']))


if __name__ == '__main__':
    unittest.main()
