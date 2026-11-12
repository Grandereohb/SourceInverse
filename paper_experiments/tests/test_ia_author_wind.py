import unittest
import numpy as np
import pandas as pd
from ia_author_wind import fill_short_gaps, aggregate_median_wind, author_wind_table


class AuthorWindTests(unittest.TestCase):
    def test_complete_short_gaps_only(self):
        x = np.array([np.nan, 0., np.nan, np.nan, 3., *([np.nan] * 6), 10., np.nan])
        y = fill_short_gaps(x)
        np.testing.assert_allclose(y[1:5], [0, 1, 2, 3])
        self.assertTrue(np.isnan(y[5:11]).all())
        self.assertTrue(np.isnan(y[[0, -1]]).all())

    def test_equal_north_wind(self):
        result = aggregate_median_wind(np.ones((2, 3)) * 2, np.zeros((2, 3)))
        np.testing.assert_allclose(result, [[0, -2], [0, -2]], atol=1e-14)

    def test_wrap_and_speed_outlier(self):
        result = aggregate_median_wind(np.array([[2., 2., 100.]]), np.array([[359., 0., 1.]]))
        np.testing.assert_allclose(result, [[0, -2]], atol=1e-14)

    def test_r_angle_conversion(self):
        angles = np.array([[20., 45., 80.], [180., 210., 240.]])
        speeds = np.array([[2., 3., 4.], [3., 4., 5.]])
        median_angle = np.arctan2(np.median(np.sin(np.deg2rad(angles)), axis=1),
                                  np.median(np.cos(np.deg2rad(angles)), axis=1))
        theta = np.deg2rad(270.) - median_angle
        expected = np.median(speeds, axis=1)[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])
        np.testing.assert_allclose(aggregate_median_wind(speeds, angles), expected, atol=1e-14)

    def test_opposing_directions_undefined(self):
        result = aggregate_median_wind(np.ones((1, 2)), np.array([[0., 180.]]))
        self.assertTrue(np.isnan(result).all())

    def test_input_not_mutated(self):
        speed = np.array([[1.], [np.nan], [3.]])
        before = speed.copy()
        aggregate_median_wind(speed, np.zeros_like(speed))
        np.testing.assert_array_equal(speed, before)

    def test_duplicate_first_and_minute_end_timestamp(self):
        raw = pd.DataFrame(dict(name=['E', 'E', 'NW', 'SW'],
            time=['2022-05-08 00:01:00'] * 4,
            **{'wind.speed': [2., 100., 2., 2.], 'wind.direction': [0., 180., 0., 0.]}))
        result = author_wind_table(raw)
        self.assertEqual(str(result.index[0]), '2022-05-08 00:00:00+00:00')
        np.testing.assert_allclose(result.to_numpy(), [[0., -2.]], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
