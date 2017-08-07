#!/usr/bin/env bash
#
# Wrapper for zip files
# usage: zip-wrapper.sh <zip>


ZIPDIR="$(mktemp -d)"
SIMDIR="$(mktemp -d)"
trap "rm -rf '$ZIPDIR' '$SIMDIR'" INT TERM EXIT

unzip "$1" -d "$ZIPDIR" &>/dev/null
BATCH="$ZIPDIR/$(dirname ${1%%.zip})/script/batch"
while read CONF; do
    <<< "$CONF" jq '.assignment = (.assignment | with_entries(.key as $key | .value = (.value | to_entries | map(.key as $key | [range(.value) | $key]) | add)))' > "$SIMDIR/simulation_spec.json"
    "$BATCH" "$SIMDIR" 1
    jq -c . "$SIMDIR/"*observation*.json
    rm "$SIMDIR/"*observation*.json
done
