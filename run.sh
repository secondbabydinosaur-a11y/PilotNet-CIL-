#!/usr/bin/env bash
# 이 PC의 사용자 CUDA 11 / 시스템 CUDA 12 패키지 경로 충돌을 실행별로 우회한다.
# 설치 파일이나 전역 환경은 바꾸지 않는다. 정상적인 다른 환경에서는 python3를 그대로 쓴다.
set -euo pipefail
PILOTNET_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PILOTNET_ROOT"
if python3 -c 'import torch' >/dev/null 2>&1; then
    exec python3 "$@"
elif [[ -d /usr/local/lib/python3.8/dist-packages/nvidia ]]; then
    exec env PYTHONPATH="/usr/local/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
else
    exec python3 "$@"
fi
