#!/bin/bash

# 1. Parameters
SCENARIOS=("ds4" "ds5" "ds6")
BASE_DIR="/mnt/hgfs/Database/TEXBAT"
WORK_DIR="$HOME/gnss-sdr/processed_Mao_TEXBAT"

cd $WORK_DIR

for SCENE in "${SCENARIOS[@]}"
do
    echo "--------------------------------------------------"
    echo "Starting _r2 processing: $SCENE"
    echo "--------------------------------------------------"
    
    BIN_FILE="$BASE_DIR/${SCENE}.bin"
    CONF_FILE="temp_${SCENE}_r2.conf"
    
    if [ ! -f "$BIN_FILE" ]; then
        echo "[ERROR]: $BIN_FILE not found, skipping."
        continue
    fi
    
    # 2. Temporary configuration without fixed-satellite lines
    cp texbat_base.conf $CONF_FILE
    sed -i "/^Channel[0-9].satellite=.*/d" $CONF_FILE
    
    # 3. Input file
    sed -i "s|^SignalSource.filename=.*|SignalSource.filename=$BIN_FILE|g" $CONF_FILE
    
    # 4. Five fixed satellites per scenario (channels 0-4)
    if [ "$SCENE" == "ds4" ]; then
        PRNS=(3 23 6 7 13)
    elif [ "$SCENE" == "ds5" ]; then
        PRNS=(6 9 15 18 21)
    elif [ "$SCENE" == "ds6" ]; then
        PRNS=(6 9 15 18 22)
    fi

    echo "[INFO]: Fixing 5 satellites: ${PRNS[*]}"
    for i in {0..4}
    do
        echo "Channel$i.satellite=${PRNS[$i]}" >> $CONF_FILE
    done
    echo "[INFO]: Channels 5-7 remain in Blind Search mode."

    # 5. Clear the dump folder and run the receiver
    rm -rf ./temp_dump/*
    mkdir -p ./temp_dump
    
    gnss-sdr --config_file=$CONF_FILE
    
    # 6. Move the output
    python3 move_texbat_data.py "${SCENE}_r2"
    
    # Remove the temporary configuration
    rm $CONF_FILE
    
    echo "[SUCCESS]: $SCENE _r2 completed."
    echo ""
done

echo "All _r2 tasks finished."
