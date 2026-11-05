import unittest

from learning.assessment import recovery_metrics, telemetry_metrics


class AssessmentTests(unittest.TestCase):
    def test_recovery_requires_motion_both_errors_and_continuous_hold(self):
        def row(t, station, cte=0., heading=0.):
            return dict(elapsed_s=t, route_s_m=station, cte_m=cte,
                        heading_error_deg=heading, reason="drive")
        # 정차 중 작은 오차로 복귀를 인정하지 않고, 잠깐 복귀한 뒤 재이탈한 경우도 구분한다.
        rows = [row(0, 100), row(2, 100), row(3, 103), row(3.5, 105, heading=3),
                row(4, 107), row(4.5, 109), row(5.01, 111)]
        result = recovery_metrics(rows, 100)
        self.assertTrue(result["recovered"])
        self.assertEqual(result["recovery_time_s"], 4)
        self.assertEqual(result["recovery_distance_m"], 7)
        self.assertFalse(recovery_metrics(rows[:-1], 100)["recovered"])

    def test_distance_terminal_error_and_new_frame_latency(self):
        rows = [dict(x_m=0, y_m=0, cte_m=.1, heading_error_deg=0, speed_kph=10, inference_ms=""),
                dict(x_m=3, y_m=4, cte_m=.3, heading_error_deg=2, speed_kph=12, inference_ms="2")]
        result = telemetry_metrics(rows, terminal_cte=1.1)
        self.assertEqual(result["distance_driven_m"], 5.)
        self.assertEqual(result["max_abs_cte_m"], 1.1)
        self.assertAlmostEqual(result["cte_rmse_m"], .05**.5)
        self.assertEqual(result["inference_samples"], 1)
        self.assertEqual(result["inference_mean_ms"], 2.)


if __name__ == "__main__":
    unittest.main()
