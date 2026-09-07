#!/bin/bash
# End-to-end preparation of the SAGA RL training data from the released
# M-Drama annotations. See data/rl/README.md for the manual (step-by-step) flow.
#
# Usage:
#   bash data/rl/run.sh --video_dir /path/to/clips \
#        [--cap2graph /path/to/gt_graphs.json] \
#        [--video_map  /path/to/video_map.json]
#
# Prerequisites:
#   * pip install -r requirements.txt   (needs: datasets, pandas, pyarrow)
#   * the M-Drama repo is gated -> huggingface-cli login (or export HF_TOKEN=...)
#   * the clips themselves: M-Drama ships annotations only. Either let
#     sft/data/download_videos.py fetch them into data/videos (named
#     {video_id}.mp4, no --video_map needed), or point --video_dir at an
#     existing clip library and let this script resolve the names.
#   * gt_graphs.json for the graph reward (without it gt_graph stays empty):
#     python -c "from huggingface_hub import hf_hub_download; \
#       print(hf_hub_download('yixin1121/M-Drama', 'gt_graphs.json', repo_type='dataset'))"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VIDEO_DIR=""
CAP2GRAPH=""
VIDEO_MAP=""
STAGE="SFT+RL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video_dir)  VIDEO_DIR="$2";  shift 2 ;;
    --cap2graph)  CAP2GRAPH="$2";  shift 2 ;;
    --video_map)  VIDEO_MAP="$2";  shift 2 ;;
    --stage)      STAGE="$2";      shift 2 ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${VIDEO_DIR}" ]]; then
  echo "error: --video_dir is required" >&2
  exit 1
fi

# ---------------------------------------------------------------- 1. annotations
echo "==> [1/3] downloading M-Drama annotations (stage=${STAGE})"
python data/rl/prepare_rl_data.py download --stage "${STAGE}"

# ------------------------------------------------------------------ 2. video map
# Only needed when the clips are not already named {video_id}.mp4. Detect the
# {title}_{HH:MM:SS}-{HH:MM:SS}.mp4 library layout; skip this step otherwise.
if [[ -z "${VIDEO_MAP}" ]]; then
  if compgen -G "${VIDEO_DIR}/*_??:??:??-??:??:??.mp4" > /dev/null; then
    echo "==> [2/3] resolving video_id -> clip path (timestamp matching)"
    python data/rl/prepare_rl_data.py video-map \
        --video_dir "${VIDEO_DIR}" \
        --mode timestamp
  else
    echo "==> [2/3] no {title}_{start}-{end}.mp4 clips found; assuming {video_id}.mp4, skipping video-map"
  fi
fi

# ------------------------------------------------------------------ 3. parquet
echo "==> [3/3] building data/rl/mdrama_rl_graph/train.parquet"
BUILD_ARGS=(--video_dir "${VIDEO_DIR}")
[[ -n "${CAP2GRAPH}" ]] && BUILD_ARGS+=(--cap2graph "${CAP2GRAPH}")
[[ -n "${VIDEO_MAP}" ]] && BUILD_ARGS+=(--video_map "${VIDEO_MAP}")
python data/rl/prepare_rl_data.py build "${BUILD_ARGS[@]}"

echo "==> done"
