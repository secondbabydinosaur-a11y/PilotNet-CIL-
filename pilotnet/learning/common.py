"""체크포인트, 재현 정보, 회귀 지표를 한 곳에서 관리한다."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch

from data_utils import BASE, read_json, write_json
from model import PilotNet


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_contract():
    cfg = read_json(BASE / "config.json")
    report = read_json(BASE / "dataset/paired/summary.json")
    model = report["model"]
    return {
        "preprocessing": cfg["preprocessing"],
        "camera_common": cfg["camera_common"],
        "center_camera": cfg["cameras"]["center"],
        "wheelbase_m": cfg["vehicle"]["wheelbase_m"],
        "steer_left_sign": model["steer_left_sign"],
        "road_wheel_gain": model["road_wheel_gain"],
        "target": "y_curvature_1pm; left negative, right positive",
        "input": "BCHW YUV float32 0..255; normalization inside model",
        "source_sha256": {name: file_hash(BASE / name) for name in (
            "dataset/splits/groups.json", "dataset/paired/summary.json",
            "dataset/paired/train.csv", "dataset/paired/val.csv", "dataset/paired/test.csv")},
    }


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = dict(schema_version=1, architecture=PilotNet.architecture,
                   model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   **metadata)
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def load_checkpoint(path, device, check_data=False):
    # 프로젝트에서 생성한 state_dict와 기본 자료형만 저장한다.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("architecture") != PilotNet.architecture:
        raise ValueError("unsupported PilotNet checkpoint")
    if check_data and payload["data_contract"] != data_contract():
        raise ValueError("dataset/config differs from the checkpoint data contract")
    model = PilotNet()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def regression_metrics(target, prediction, wheelbase=3.):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(target) or target.shape != prediction.shape:
        raise ValueError("nonempty matching arrays required")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("non-finite regression values")
    error = prediction - target
    sst = np.sum((target - target.mean()) ** 2)
    angle_error = np.degrees(np.arctan(-prediction * wheelbase)
                             - np.arctan(-target * wheelbase))
    return {"samples": len(target), "mse_1pm2": float(np.mean(error ** 2)),
            "rmse_1pm": float(np.sqrt(np.mean(error ** 2))),
            "mae_1pm": float(np.mean(np.abs(error))),
            "p95_abs_error_1pm": float(np.percentile(np.abs(error), 95)),
            "r2": float(1 - np.sum(error ** 2) / sst) if sst > 1e-15 else None,
            "bicycle_steer_mae_deg": float(np.mean(np.abs(angle_error)))}


def environment_info():
    return {"torch": str(torch.__version__), "numpy": np.__version__,
            "opencv": cv2.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def json_print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
