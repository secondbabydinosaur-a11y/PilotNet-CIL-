"""GitHub에 보관한 가중치/계획을 원래 로컬 runs 경로로 복원한다. 학습/주행 없음."""
import shutil

from data_utils import BASE
from learning.common import file_hash


def main():
    pairs = [(BASE / "docs/evaluation_protocol.json", BASE / "runs/github_protocol.json")]
    for mode in ("center", "three"):
        for name in ("best.pt", "settings.json", "summary.json", "history.csv", "learning_curve.png"):
            folder = "center_camera" if mode == "center" else "three_cameras"
            pairs.append((BASE / "checkpoints" / folder / name,
                          BASE / ("runs/pilotnet_" + mode + "_v1") / name))
    # 다른 작업 결과가 이미 있으면 덮어쓰지 않고 먼저 중단한다.
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and file_hash(source) != file_hash(target):
            raise ValueError("existing local artifact differs; preserve it: " + str(target))
    for source, target in pairs:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
    print("Archived baseline models/protocol restored or already identical. No training or driving performed.")


if __name__ == "__main__":
    main()
