#!/bin/bash

for dir in tmp_*; do
    incident_idx=$(echo "$dir" | cut -d'_' -f2)
    if [ -f "$dir/0.cont" ] && [ ! -f "output/${incident_idx}.cont" ]; then
        echo "Copying ${incident_idx}.cont"
        cp "$dir/0.cont" "output/${incident_idx}.cont"
    fi
done
