#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT=""
OUTPUT_ROOT=""
KEYINT=10
MAX_WIDTH=640
CRF=23
PRESET="veryfast"
JOBS="$(nproc)"
DRY_RUN=0
SKIP_EXISTING=1

usage() {
  cat <<'EOF'
Batch re-encode videos for faster random access:
- shorter GOP (fixed keyframe interval)
- lower resolution (max width, keep aspect ratio)

Usage:
  reencode_videos_short_gop.sh \
    --input-root /path/to/videos \
    --output-root /path/to/videos_reencoded \
    [--keyint 10] [--max-width 640] [--crf 23] [--preset veryfast] [--jobs 8] [--dry-run]

Notes:
  - Keeps relative directory structure under output root.
  - Encodes with H.264 (libx264), fixed GOP, no B-frames, yuv420p.
  - By default, skips files that already exist in output root.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --keyint)
      KEYINT="$2"
      shift 2
      ;;
    --max-width)
      MAX_WIDTH="$2"
      shift 2
      ;;
    --crf)
      CRF="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_ROOT" || -z "$OUTPUT_ROOT" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input root does not exist: $INPUT_ROOT" >&2
  exit 1
fi

if [[ "$(realpath "$INPUT_ROOT")" == "$(realpath "$OUTPUT_ROOT" 2>/dev/null || echo "$OUTPUT_ROOT")" ]]; then
  echo "Input and output roots must be different to avoid overwriting source videos." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not found in PATH." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mapfile -d '' FILES < <(find "$INPUT_ROOT" -type f -name '*.mp4' -print0 | sort -z)
TOTAL="${#FILES[@]}"

if [[ "$TOTAL" -eq 0 ]]; then
  log "No .mp4 files found under $INPUT_ROOT"
  exit 0
fi

VF="scale=w='min(iw,${MAX_WIDTH})':h=-2:flags=lanczos"
FAILED=0
COUNT=0

transcode_one() {
  local src="$1"
  local rel dst
  rel="${src#"$INPUT_ROOT"/}"
  dst="${OUTPUT_ROOT}/${rel}"

  mkdir -p "$(dirname "$dst")"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$dst" ]]; then
    log "skip existing: $rel"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: $rel"
    return 0
  fi

  ffmpeg -hide_banner -loglevel error -y \
    -i "$src" \
    -map 0:v:0 \
    -map_metadata -1 \
    -c:v libx264 \
    -preset "$PRESET" \
    -crf "$CRF" \
    -pix_fmt yuv420p \
    -g "$KEYINT" \
    -x264-params "keyint=${KEYINT}:min-keyint=${KEYINT}:scenecut=0" \
    -bf 0 \
    -vf "$VF" \
    -movflags +faststart \
    -write_tmcd 0 \
    "$dst"
}

log "Start re-encode: total=$TOTAL keyint=$KEYINT max_width=$MAX_WIDTH crf=$CRF preset=$PRESET jobs=$JOBS"

for src in "${FILES[@]}"; do
  ((COUNT += 1))
  rel="${src#"$INPUT_ROOT"/}"
  log "queue [$COUNT/$TOTAL] $rel"
  transcode_one "$src" &

  while [[ "$(jobs -r -p | wc -l)" -ge "$JOBS" ]]; do
    if ! wait -n; then
      FAILED=1
    fi
  done
done

while [[ "$(jobs -r -p | wc -l)" -gt 0 ]]; do
  if ! wait -n; then
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  log "Completed with failures. Please check logs above."
  exit 1
fi

log "Completed successfully."
