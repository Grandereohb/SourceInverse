import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_paper_industrial_cases as preparation


class IndustrialPreparationTests(unittest.TestCase):
    def test_parser_scientific_notation_unknown_codes_and_full_match(self):
        value=preparation.parse_measurement('1.25e-3(N)')
        self.assertEqual(value['status'],'N')
        self.assertAlmostEqual(value['point_value'],.00125)
        self.assertEqual(preparation.parse_measurement('bad123')['state'],'unparsed')
        self.assertIsNone(preparation.parse_measurement('--')['point_value'])
        self.assertEqual(preparation.parse_measurement('−0.3(M)')['point_value'],-.3)
        relayed=preparation.parse_measurement('0.01(N回传)')
        self.assertEqual(relayed['point_value'],.01)
        self.assertEqual(relayed['status'],'N回传')

    def test_reported_bounds_are_not_replaced_with_point_values(self):
        value=preparation.parse_measurement('≤0.02(E)')
        self.assertEqual(value['reported_value'],.02)
        self.assertEqual(value['state'],'bounded')
        self.assertIsNone(value['point_value'])

    def test_workbook_reader_preserves_opaque_tags_and_duplicate_provenance(self):
        class Workbook:
            sheet_names=['A(带标识)']
            def __enter__(self):
                return self
            def __exit__(self,*args):
                return False
        table=pd.DataFrame({'时间':['标准单位','2026-05-01 00:00:00','2026-05-01 00:00:00'],
                            'P':['μg/m³','0.01(N回传)','0.03(M)'],
                            '风速':['m/s',0.,0.],'风向':['°',10.,10.]})
        with patch.object(pd,'ExcelFile',return_value=Workbook()), \
             patch.object(pd,'read_excel',return_value=table):
            tables,units,raw,inventory=preparation.read_monitor('unused.xlsx',{'P'})
        self.assertAlmostEqual(tables['A']['P'].iloc[0],.02)
        self.assertEqual(units['A','P'],'μg/m³')
        self.assertEqual(inventory.iloc[0].duplicate_rows,1)
        values=raw[raw.variable=='P']
        self.assertEqual(values.status.tolist(),['N回传','M'])
        self.assertEqual(values.source_excel_row.tolist(),[3,4])

    def test_merging_is_within_pollutant_and_transitive(self):
        rows=pd.DataFrame(dict(station=['A']*4,pollutant=['P','P','P','Q'],
            time=pd.to_datetime(['2026-05-03 00:00','2026-05-03 12:00',
                                 '2026-05-04 01:00','2026-05-03 01:00']),
            concentration=[50.,70.,60.,90.]))
        events=preparation.merge_catalog_events(rows)
        self.assertEqual(len(events),3)
        p=[v for v in events if v['pollutant']=='P']
        self.assertEqual(p[0]['catalog_row_count'],2)
        self.assertEqual(p[0]['catalog_peak'],70.)
        self.assertEqual(p[0]['peak_time'],pd.Timestamp('2026-05-03 12:00'))

    def test_shared_baseline_support_is_not_counted_independent(self):
        cases=[dict(case_id='a',window_start='2026-05-03 00:00',window_end='2026-05-03 12:00'),
               dict(case_id='b',window_start='2026-05-03 17:00',window_end='2026-05-03 23:00'),
               dict(case_id='c',window_start='2026-05-05 00:00',window_end='2026-05-05 12:00')]
        clusters=preparation.temporal_clusters(cases)
        self.assertEqual(clusters['a'],clusters['b'])
        self.assertNotEqual(clusters['a'],clusters['c'])

    def test_partial_missing_observations_remain_masked_and_signed(self):
        rows=pd.DataFrame(dict(station=['A'],pollutant=['P'],
            time=[pd.Timestamp('2026-05-03 12:00')],concentration=[30.]))
        event=preparation.merge_catalog_events(rows)[0]
        labels=pd.date_range('2026-05-03 00:00',periods=25,freq='h')
        tables={s:pd.DataFrame(dict(P=np.full(25,5.),风速=np.zeros(25),风向=np.zeros(25)),index=labels)
                for s in ('A','B','C')}
        tables['A'].loc[pd.Timestamp('2026-05-03 12:00'),'P']=30.
        tables['B'].loc[pd.Timestamp('2026-05-03 12:00'),'P']=np.nan
        tables['C'].loc[pd.Timestamp('2026-05-03 12:00'),'P']=4.
        units={(s,k):v for s in tables for k,v in (('P','μg/m³'),('风速','m/s'),('风向','°'))}
        raw=pd.DataFrame([dict(station='A',variable='P',time='2026-05-03 12:00:00',
                              point_value=30.,status='N',state='point_numeric')])
        coords={s:(117.,39.) for s in tables}
        meta,data=preparation.prepare_case(event,tables,units,coords,raw)
        self.assertTrue(meta['eligible'])
        self.assertEqual(meta['point_observation_count'],38)
        self.assertEqual(data['residual_concentration'][6,2],-1.)
        self.assertFalse(data['observation_mask'][6,1])
        self.assertTrue(np.isnan(data['normalized_concentration'][6,1]))
        tables['A'].loc[pd.Timestamp('2026-05-03 10:00'),'风速']=np.nan
        meta,data=preparation.prepare_case(event,tables,units,coords,raw)
        self.assertFalse(meta['eligible'])
        self.assertIsNone(data)
        self.assertIn('incomplete_or_invalid_hourly_wind_no_interpolation',meta['rejection_reasons'])


if __name__=='__main__':
    unittest.main()
