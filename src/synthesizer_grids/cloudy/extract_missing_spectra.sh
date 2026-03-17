#!/bin/bash

for dir in tmp_*; do
    incident_idx=$(echo "$dir" | cut -d'_' -f2)
    if [ -f "$dir/0.cont" ] && [ ! -f "spectra/spectra_${incident_idx}.txt" ]; then
        echo "Creating spectra_${incident_idx}.txt"
        tail -n +2 "$dir/0.cont" | awk '{print $1, $3, $4, $7}' > "spectra/spectra_${incident_idx}.txt"
    fi
done
