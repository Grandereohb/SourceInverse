import math
import unittest

from scripts.summarize_reliability_audit_test import summarize, wilson_lower_bound


class ReliabilityAuditSummaryTests(unittest.TestCase):
    def test_one_sided_wilson_lower_bound(self):
        self.assertEqual(wilson_lower_bound(0, 10), 0.0)
        self.assertAlmostEqual(wilson_lower_bound(5, 10), 0.2692718, places=6)
        with self.assertRaises(ValueError):
            wilson_lower_bound(11, 10)

    def test_joint_rule_and_release_stratum_are_preserved(self):
        primary = ["gaussian_dynamic", "adr_constant"]
        protocol = {
            "design_id": "test_design",
            "split": "test-OOD",
            "success_threshold_m": 500.0,
            "preregistered_analysis": {
                "primary_estimand": "selection failure among reachable events",
                "primary_methods": primary,
            },
        }

        def rows(method):
            output = []
            for index in range(20):
                failed = index < 10
                output.append(
                    {
                        "scenario_id": f"s{index}",
                        "method": method,
                        "input_id": "wind_a",
                        "position_id": "centre",
                        "physics_condition_id": "particle",
                        "release_condition_id": (
                            "constant_release" if index % 2 == 0 else "double_peak_release"
                        ),
                        "selected_success": not failed,
                        "oracle_success": True,
                        "failure_category": (
                            "selection_failure_reachable" if failed else "selected_success"
                        ),
                        "selected_localization_error_m": 600.0 if failed else 100.0,
                        "oracle_localization_error_m": 50.0,
                        "selection_regret_m": 550.0 if failed else 50.0,
                    }
                )
            return output

        result = summarize(
            protocol,
            [("surface", {"rows": rows(primary[0]) + rows(primary[1])})],
        )

        self.assertTrue(result["joint_confirmation_passed"])
        self.assertGreater(
            result["method_results"][primary[0]][
                "selection_failure_wilson_lower_bound"
            ],
            0.10,
        )
        release_levels = {
            row["level"]
            for row in result["strata"]
            if row["factor"] == "release_condition_id"
        }
        self.assertEqual(
            release_levels,
            {"constant_release", "double_peak_release"},
        )

    def test_missing_primary_method_is_rejected(self):
        protocol = {
            "design_id": "test_design",
            "split": "test-OOD",
            "success_threshold_m": 500.0,
            "preregistered_analysis": {
                "primary_estimand": "selection failure",
                "primary_methods": ["missing"],
            },
        }
        with self.assertRaises(ValueError):
            summarize(protocol, [("surface", {"rows": []})])


if __name__ == "__main__":
    unittest.main()
