#!/bin/bash

# Writes current_<scenario>.conf (texbat_base.conf with the input file of the scenario) and
# runs GNSS-SDR on it. The scenario list is as last edited; ds3 (31 March 2026) and ds7, ds8
# (9 April 2026) were processed with this script.
SCENARIOS=("ds5" "ds6")
BASE_DIR="/mnt/hgfs/Database/TEXBAT"

mkdir -p "$HOME/gnss-sdr/processed_Mao_TEXBAT/temp_dump"

for SCENE in "${SCENARIOS[@]}"
do
    echo "=================================================="
    echo "Processing $SCENE"
    echo "=================================================="
    
    BIN_FILE="$BASE_DIR/${SCENE}.bin"
    
    if [ ! -f "$BIN_FILE" ]; then
        echo "[ERROR] $BIN_FILE not found, skipping"
        continue
    fi
    
    echo "Input: $BIN_FILE"
    
    # configuration of this run: texbat_base.conf with the input file of the scenario
    sed "s|^SignalSource.filename=.*|SignalSource.filename=$BIN_FILE|g" texbat_base.conf > "current_$SCENE.conf"
    
    # run the receiver
    gnss-sdr --config_file="current_$SCENE.conf"
    
    # move the .dat output to the shared folder
    python3 move_texbat_data.py "$SCENE"
    
    echo "$SCENE done"
    echo ""
done

echo "All scenarios processed."
