"""고정 모델의 미학습 구간 수집/평가와 헤딩 복귀 시험을 순차 실행한다."""
import subprocess

from data_utils import BASE
from tools.collection.collect_unseen import verify_frozen


def main():
    verify_frozen()
    commands = (
        ("-m", "tools.collection.collect_unseen"),
        ("-m", "tools.evaluation.evaluate_unseen"),
        ("-m", "tools.evaluation.run_matrix", "--courses", "U1", "U2", "--repeats", "3",
         "--record-screen", "--tag", "unseen"),
        ("-m", "tools.evaluation.run_matrix", "--courses", "U1", "U2", "--conditions",
         "heading_left", "heading_right", "--record-screen", "--tag", "unseen"),
        ("-m", "tools.evaluation.build_github_artifacts"),
    )
    for args in commands:
        print("RUN", " ".join(args), flush=True)
        subprocess.run(["bash", str(BASE / "run.sh"), *args], cwd=str(BASE), check=True)
        verify_frozen()


if __name__ == "__main__":
    main()
