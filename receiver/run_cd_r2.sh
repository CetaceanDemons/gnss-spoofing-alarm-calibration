#!/bin/bash
# run_cd_r2.sh
# TEXBAT cleanDynamic, processed with the procedure of run_r2.sh and texbat_base.conf.
# Differences from run_r2.sh: scenario cleanDynamic; fixed satellites 6 9 15 18 21 22
# (union of ds5 and ds6) on channels 0-5; channels 6-7 search freely.
BASE_DIR="/mnt/hgfs/Database/TEXBAT"
WORK_DIR="$HOME/gnss-sdr/processed_Mao_TEXBAT"
SCENE="cleanDynamic"
TAG="${SCENE}_r2"
BIN_FILE="$BASE_DIR/${SCENE}.bin"
CONF_FILE="temp_${TAG}.conf"
OUT_DIR="$BASE_DIR/processed_Mao/$TAG"
PRNS=(6 9 15 18 21 22)

cd "$WORK_DIR" || exit 1
[ -f "$BIN_FILE" ] || { echo "[ERROR] $BIN_FILE not found"; exit 1; }
[ -e "$OUT_DIR" ] && { echo "[ERROR] $OUT_DIR exists: move it away first (the modified receiver appends to its output files)"; exit 1; }

cp texbat_base.conf "$CONF_FILE"
sed -i "/^Channel[0-9].satellite=.*/d" "$CONF_FILE"
sed -i "s|^SignalSource.filename=.*|SignalSource.filename=$BIN_FILE|g" "$CONF_FILE"
for i in "${!PRNS[@]}"; do echo "Channel$i.satellite=${PRNS[$i]}" >> "$CONF_FILE"; done
echo "[INFO] fixed satellites: ${PRNS[*]} (channels 0-5); channels 6-7 search freely"

# record the configuration used and the checksums of the base configuration, receiver and input
cp "$CONF_FILE" "${TAG}_used.conf"
{ md5sum texbat_base.conf "${TAG}_used.conf" "$(command -v gnss-sdr)"
  ls -l --time-style=long-iso texbat_base.conf "$(command -v gnss-sdr)" "$BIN_FILE"
  gnss-sdr --version 2>&1 | head -1; } > "${TAG}_provenance.txt"

rm -rf ./temp_dump/*
mkdir -p ./temp_dump
date '+[START] %F %T' | tee "${TAG}_run.log"
gnss-sdr --config_file="$CONF_FILE" 2>&1 | tee -a "${TAG}_run.log"
date '+[END] %F %T' | tee -a "${TAG}_run.log"

python3 move_texbat_data.py "$TAG"
cp "${TAG}_used.conf" "${TAG}_provenance.txt" "${TAG}_run.log" "$OUT_DIR/"
rm "$CONF_FILE"
ls -l "$OUT_DIR"
echo "[SUCCESS] $TAG done"
