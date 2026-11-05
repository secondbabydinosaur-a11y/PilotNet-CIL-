"""닫힌 루프 로그 평가. 주행 제어 및 모델 선택에는 사용하지 않는다."""
import numpy as np


def recovery_metrics(rows, start_station):
    """3m 이상 진행 후 작은 횡/헤딩 오차를 1초 유지한 첫 구간을 찾는다."""
    candidate = None
    for row in rows:
        t = float(row["elapsed_s"])
        progress = float(row["route_s_m"]) - start_station
        good = (progress >= 3. and abs(float(row["cte_m"])) <= .15
                and abs(float(row["heading_error_deg"])) <= 2.
                and row["reason"] == "drive")
        if not good:
            candidate = None
        elif candidate is None:
            candidate = (t, progress)
        elif t - candidate[0] >= 1.:
            return dict(recovered=True, recovery_time_s=candidate[0],
                        recovery_distance_m=candidate[1], confirmed_at_s=t)
    return dict(recovered=False, recovery_time_s=None, recovery_distance_m=None, confirmed_at_s=None)


def telemetry_metrics(rows, terminal_cte=None):
    cte = np.array([float(r["cte_m"]) for r in rows])
    if not len(cte):
        return {}
    observed = np.r_[cte, terminal_cte] if terminal_cte is not None else cte
    inference = [float(r["inference_ms"]) for r in rows if r.get("inference_ms") not in ("", None)]
    return dict(cte_rmse_m=float(np.sqrt(np.mean(cte ** 2))),
        cte_mae_m=float(np.mean(np.abs(cte))), p95_abs_cte_m=float(np.percentile(np.abs(cte), 95)),
        max_abs_cte_m=float(max(np.abs(observed))),
        max_abs_heading_deg=float(max(abs(float(r["heading_error_deg"])) for r in rows)),
        mean_speed_kph=float(np.mean([float(r["speed_kph"]) for r in rows])),
        max_speed_kph=float(max(float(r["speed_kph"]) for r in rows)),
        distance_driven_m=float(np.sum(np.hypot(np.diff([float(r["x_m"]) for r in rows]),
                                                 np.diff([float(r["y_m"]) for r in rows])))),
        inference_samples=len(inference), inference_mean_ms=float(np.mean(inference)) if inference else None,
        inference_p95_ms=float(np.percentile(inference, 95)) if inference else None)
