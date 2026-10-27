import unittest

from audit_metec_public_sample import longest_missing


class MetecSampleAuditTests(unittest.TestCase):
    def test_missing_wind_run_is_not_total_count(self):
        self.assertEqual(longest_missing([True,False,False,True,False]),2)
        self.assertEqual(longest_missing([True,True]),0)
        self.assertEqual(longest_missing([False,False,False]),3)


if __name__ == '__main__':
    unittest.main()
