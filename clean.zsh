#!/bin/zsh

set -euo pipefail
setopt extended_glob

DATA_DIR="./data"
LOG_DIR="./logs"

KEEP_FILES=(.gitkeep visualise.py README.md main.py pyproject.toml uv.lock .python-version)
KEEP_DIRS=(.venv)

delete_dataset_dir() {
  local dir=$1
  if [[ -d $dir ]]; then
    echo "Removing dataset directory: $dir"
    rm -rf -- "$dir"
  fi
}

clean_data() {
  [[ -d $DATA_DIR ]] || { echo "Data directory not found: $DATA_DIR" >&2; return; }

  for entry in "$DATA_DIR"/*; do
    [[ -e $entry ]] || continue
    local base=${entry:t}
    if [[ -f $entry ]]; then
      case $base in
        (${^KEEP_FILES})
          continue
          ;;
        (probed_mu_T_*.csv)
          echo "Removing sampled CSV: $entry"
          rm -f -- "$entry"
          ;;
        (*)
          continue
          ;;
      esac
    elif [[ -d $entry ]]; then
      case $base in
        (${^KEEP_DIRS})
          continue
          ;;
        (<->.<->_<->.<->)
          delete_dataset_dir "$entry"
          ;;
        (*)
          continue
          ;;
      esac
    fi
  done
}

clean_logs() {
  [[ -d $LOG_DIR ]] || return
  find "$LOG_DIR" -type f -mindepth 1 -delete
}

clean_data
clean_logs

echo "Cleanup complete."
