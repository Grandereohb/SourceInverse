import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_piecewise_segments(t_values, segment_length):
    t_values = np.asarray(t_values, dtype=float)
    n_times = len(t_values)
    segment_length = max(1, int(segment_length))
    n_segments = max(1, int(math.ceil(n_times / segment_length)))

    segment_ids = np.minimum(np.arange(n_times) // segment_length, n_segments - 1)
    breaks = []
    for seg_id in range(1, n_segments):
        prev_idx = np.where(segment_ids == seg_id - 1)[0][-1]
        next_idx = np.where(segment_ids == seg_id)[0][0]
        breaks.append(0.5 * (t_values[prev_idx] + t_values[next_idx]))
    return n_segments, np.asarray(breaks, dtype=np.float32), segment_ids


def configure_model_q(
    model,
    q_mode,
    t_values,
    segment_length,
    device,
):
    q_mode = (q_mode or "neural").lower()
    if q_mode == "smooth_time":
        if hasattr(model, "configure_smooth_time_q"):
            model.configure_smooth_time_q(
                torch.tensor(t_values, dtype=torch.float32, device=device)
            )
        else:
            raise AttributeError("Model does not support smooth_time Q mode.")
        model.to(device)
        return {
            "mode": "smooth_time",
            "n_segments": 0,
            "breaks": np.asarray([], dtype=np.float32),
            "segment_ids": np.full(len(t_values), -1, dtype=int),
        }

    if q_mode != "piecewise":
        if hasattr(model, "configure_neural_q"):
            model.configure_neural_q(
                torch.tensor(t_values, dtype=torch.float32, device=device)
            )
        else:
            model.q_mode = "neural"
        model.to(device)
        return {
            "mode": "neural",
            "n_segments": 0,
            "breaks": np.asarray([], dtype=np.float32),
            "segment_ids": np.full(len(t_values), -1, dtype=int),
        }

    n_segments, breaks, segment_ids = build_piecewise_segments(
        t_values=t_values,
        segment_length=segment_length,
    )
    model.configure_piecewise_q(
        n_segments=n_segments,
        segment_breaks=torch.tensor(breaks, dtype=torch.float32, device=device),
    )
    model.to(device)
    return {
        "mode": "piecewise",
        "n_segments": n_segments,
        "breaks": breaks,
        "segment_ids": segment_ids,
    }


def compute_q_losses(model):
    if hasattr(model, "q_regularization"):
        return model.q_regularization()
    zero = next(model.parameters()).sum() * 0.0
    return zero, zero


def export_q_time_series(
    model,
    t_values,
    time_labels,
    output_dir,
    q_min=None,
    q_max=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_values = np.asarray(t_values, dtype=np.float32)
    if time_labels is None:
        time_labels = t_values

    with torch.no_grad():
        device = next(model.parameters()).device
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device).view(-1, 1)
        q_tensor = model.Q(t_tensor).detach().cpu().view(-1)
        if q_min is not None or q_max is not None:
            q_tensor = torch.clamp(
                q_tensor,
                min=float(q_min) if q_min is not None else None,
                max=float(q_max) if q_max is not None else None,
            )
        q_values = q_tensor.numpy()
        logq_values = np.log(np.maximum(q_values, 1e-12))

    q_mode = getattr(model, "q_mode", "neural")
    if q_mode == "piecewise" and model.logQ_segments is not None:
        breaks = model.q_segment_breaks.detach().cpu().numpy()
        segment_ids = np.searchsorted(breaks, t_values, side="right")
    else:
        segment_ids = np.full(len(t_values), -1, dtype=int)

    q_df = pd.DataFrame(
        {
            "time": [str(t) for t in time_labels],
            "q_mode": q_mode,
            "segment_id": segment_ids,
            "Q": q_values,
            "logQ": logq_values,
        }
    )
    q_df.to_csv(output_dir / "q_time_series.csv", index=False, encoding="utf-8-sig")

    seg_rows = []
    if q_mode == "piecewise":
        for seg_id in sorted(set(segment_ids.tolist())):
            mask = segment_ids == seg_id
            seg_rows.append(
                {
                    "q_mode": q_mode,
                    "segment_id": int(seg_id),
                    "start_time": str(np.asarray(time_labels, dtype=object)[mask][0]),
                    "end_time": str(np.asarray(time_labels, dtype=object)[mask][-1]),
                    "Q": float(np.mean(q_values[mask])),
                    "logQ": float(np.mean(logq_values[mask])),
                }
            )
    else:
        seg_rows.append(
            {
                "q_mode": q_mode,
                "segment_id": -1,
                "start_time": str(np.asarray(time_labels, dtype=object)[0]),
                "end_time": str(np.asarray(time_labels, dtype=object)[-1]),
                "Q": float(np.mean(q_values)),
                "logQ": float(np.mean(logq_values)),
            }
        )
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(output_dir / "q_segments.csv", index=False, encoding="utf-8-sig")
    return q_df, seg_df
