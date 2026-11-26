"""The expert's local route is a training target for the visual path predictor."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from dataset import BASE, CAMERAS, CILDataset
from path_model import PATH_DISTANCES_M
from prepare import rotation, virtual_pose


class CILPathDataset(CILDataset):
    def __init__(self, csv_path, split="train", cameras=CAMERAS, augment=True, root_dir=BASE):
        cameras = tuple(cameras)
        super().__init__(csv_path, split=split, cameras=cameras, augment=augment, root_dir=root_dir)
        with (self.root / csv_path).open(newline="", encoding="utf-8-sig") as stream:
            rows = [row for row in csv.DictReader(stream) if row["camera"] in cameras]
        if len(rows) != len(self.samples):
            raise ValueError("path target count does not match CIL samples")

        sessions = {}
        coordinates, validity = [], []
        distance = np.asarray(PATH_DISTANCES_M)
        for row, (image, camera, *_) in zip(rows, self.samples):
            session = image.parents[2]
            if self.root / Path(row["image_path"]) != image or session.name != row["session_id"]:
                raise ValueError("path target and image session differ")
            if session not in sessions:
                meta = json.loads((session / "session.json").read_text())
                plan = self.root / meta["route_csv"]
                if meta["route_id"] != row["route_id"] or hashlib.sha256(plan.read_bytes()).hexdigest() != meta["route_sha256"]:
                    raise ValueError("captured demonstration route changed")
                path = np.loadtxt(plan, delimiter=",", skiprows=1, ndmin=2)
                sessions[session] = path, meta["config"]["cameras"]
            path, rig = sessions[session]
            ego = {key: float(row[key]) for key in ("x_m", "y_m", "yaw_deg")}
            pose = virtual_pose(ego, rig[camera], rig["center"])
            stations = float(row["route_s_m"]) + distance
            world = np.column_stack([np.interp(stations, path[:, 0], path[:, dimension]) for dimension in (1, 2)])
            local = (world - pose[:2]) @ rotation(pose[2])
            coordinates.append(local / distance[:, None])
            validity.append(stations <= path[-1, 0])
        self.paths = np.float32(coordinates)
        self.valid = np.asarray(validity, dtype=np.bool_)

    def path_label(self, index):
        return {"path_target": torch.from_numpy(self.paths[index].copy()),
                "path_valid": torch.from_numpy(self.valid[index].copy())}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample.update(self.path_label(index))
        return sample
