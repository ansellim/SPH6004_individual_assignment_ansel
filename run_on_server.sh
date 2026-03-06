#!/usr/bin/env bash
# =============================================================================
# run_on_server.sh
# Upload project to ewa.nus.edu.sg, run R scripts remotely, download results.
# Uses renv to reproduce the local R package environment on the server.
#
# Usage:
#   ./run_on_server.sh              # run all scripts (01–10)
#   ./run_on_server.sh 05 10        # run scripts 05 through 10 only
#   ./run_on_server.sh --download   # just download results (no run)
# =============================================================================
set -euo pipefail

SERVER="ansellim@ewa.nus.edu.sg"
REMOTE_DIR="~/SPH6004_individual"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Parse arguments ---------------------------------------------------
MODE="run"
SCRIPT_START=01
SCRIPT_END=11

if [[ "${1:-}" == "--download" ]]; then
    MODE="download"
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    SCRIPT_START=$(printf "%02d" "$((10#$1))")
    SCRIPT_END=$(printf "%02d" "$((10#${2:-$1}))")
fi

# ---------- Helper functions --------------------------------------------------
upload() {
    echo "==> Uploading project to $SERVER:$REMOTE_DIR ..."
    ssh "$SERVER" "mkdir -p $REMOTE_DIR/{data,figures,results}"

    # Upload R scripts, CSV data, RDS artefacts, and renv files
    rsync -avz --progress \
        --include='*.R' \
        --include='*.csv' \
        --include='*.Rproj' \
        --include='renv.lock' \
        --include='.Rprofile' \
        --include='renv/' \
        --include='renv/activate.R' \
        --include='renv/settings.json' \
        --include='data/***' \
        --exclude='renv/library/***' \
        --exclude='renv/staging/***' \
        --exclude='renv/sandbox/***' \
        --exclude='*.pdf' \
        --exclude='*.Rmd' \
        --exclude='*.md' \
        --exclude='*.sh' \
        --exclude='.git/***' \
        --exclude='figures/***' \
        --exclude='results/***' \
        "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

    echo "    Upload complete."
}

restore_renv() {
    echo "==> Restoring renv environment on server ..."
    ssh "$SERVER" bash -l <<'REMOTE_SCRIPT'
        cd ~/SPH6004_individual

        # Install renv if not available
        Rscript -e 'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv", repos="https://cloud.r-project.org")'

        # Restore packages from lockfile
        Rscript -e 'renv::restore(prompt=FALSE)'
REMOTE_SCRIPT
    echo "    renv restore complete."
}

run_scripts() {
    echo "==> Running scripts $SCRIPT_START through $SCRIPT_END on server ..."

    # Build the list of scripts to run in order
    local scripts=""
    for f in "$LOCAL_DIR"/[0-9][0-9]_*.R; do
        base=$(basename "$f")
        num=${base%%_*}
        if [[ "$((10#$num))" -ge "$((10#$SCRIPT_START))" && "$((10#$num))" -le "$((10#$SCRIPT_END))" ]]; then
            scripts="$scripts $base"
        fi
    done

    if [[ -z "$scripts" ]]; then
        echo "    No scripts found in range $SCRIPT_START–$SCRIPT_END."
        exit 1
    fi

    echo "    Scripts to run:$scripts"

    # Run each script sequentially via SSH
    # .Rprofile auto-activates renv, so Rscript picks up the project library
    for script in $scripts; do
        echo ""
        echo "--- Running $script ---"
        ssh "$SERVER" "cd $REMOTE_DIR && Rscript --vanilla $script" 2>&1 | tail -30
        echo "--- $script finished ---"
    done

    echo ""
    echo "==> All scripts completed."
}

download() {
    echo "==> Downloading results and figures from server ..."

    # Results CSVs
    mkdir -p "$LOCAL_DIR/results"
    rsync -avz "$SERVER:$REMOTE_DIR/results/" "$LOCAL_DIR/results/" 2>/dev/null || echo "    (no results/ files found)"

    # Figures
    mkdir -p "$LOCAL_DIR/figures"
    rsync -avz "$SERVER:$REMOTE_DIR/figures/" "$LOCAL_DIR/figures/" 2>/dev/null || echo "    (no figures/ files found)"

    # RDS model artefacts
    rsync -avz "$SERVER:$REMOTE_DIR/data/" "$LOCAL_DIR/data/" 2>/dev/null || echo "    (no new data/ files)"

    echo "    Download complete."
    echo ""
    echo "==> Local results:"
    ls -lh "$LOCAL_DIR/results/" 2>/dev/null || echo "    (empty)"
}

# ---------- Main --------------------------------------------------------------
case "$MODE" in
    run)
        upload
        run_scripts
        download
        ;;
    download)
        download
        ;;
esac

echo ""
echo "Done."
