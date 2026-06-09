#!/bin/bash
# Recover continuum files from any leftover temp dirs - e.g. jobs that ran
# cloudy and produced 0.cont but died before moving it into cont/. Run from
# the grid output directory. collect_sobol_outputs.py then reads cont/*.cont
# with synthesizer.read_continuum.

mkdir -p cont
for dir in tmp_*; do
    [ -d "$dir" ] || continue
    incident_idx=$(echo "$dir" | cut -d'_' -f2)
    if [ -f "$dir/0.cont" ] && [ ! -f "cont/${incident_idx}.cont" ]; then
        echo "Recovering cont/${incident_idx}.cont"
        mv "$dir/0.cont" "cont/${incident_idx}.cont"
    fi
done
