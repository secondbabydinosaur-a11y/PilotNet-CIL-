#!/usr/bin/env bash
# 모델 폴더를 명시해 PilotNet과 CIL의 설정·데이터·모델 import를 구분한다.
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run.sh <pilotnet|cil> <script.py | -m module | -c code> [arguments...]

Examples:
  bash run.sh pilotnet check_model.py
  bash run.sh pilotnet -m unittest discover -s tests -v

CIL scripts are added in cil/ as implementation progresses.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

MODEL_DIRECTORY="$1"
shift
case "$MODEL_DIRECTORY" in
    pilotnet|cil) ;;
    *) printf 'Unknown model: %s\n' "$MODEL_DIRECTORY" >&2; usage >&2; exit 2 ;;
esac
cd "$REPOSITORY_ROOT/$MODEL_DIRECTORY"

if [[ "$1" == *.py && ! -f "$1" ]]; then
    printf 'Python file does not exist yet: %s/%s\n' "$MODEL_DIRECTORY" "$1" >&2
    exit 2
fi

# 이 PC의 CUDA 패키지 경로 충돌은 실행 프로세스에서만 처리한다.
if python3 -c 'import torch' >/dev/null 2>&1; then
    exec python3 "$@"
elif [[ -d /usr/local/lib/python3.8/dist-packages/nvidia ]]; then
    exec env PYTHONPATH="/usr/local/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
else
    exec python3 "$@"
fi
