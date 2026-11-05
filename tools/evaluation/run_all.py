"""주행을 겹치지 않고 실행한 다음 발표용 결과물을 생성한다."""
import subprocess

from data_utils import BASE


if __name__ == "__main__":
    for arguments in (
        ("-m", "tools.evaluation.run_matrix", "--repeats", "3"),
        ("-m", "tools.evaluation.run_matrix", "--repeats", "1", "--record-screen"),
        ("-m", "tools.evaluation.make_results"),
    ):
        subprocess.run(["bash", str(BASE / "run.sh"), *arguments], cwd=str(BASE), check=True)
