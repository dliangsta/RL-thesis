#!/bin/bash
output_data=output_data.tar.gz
save_dir=${1:-"./"}

# Tar tmp output.
tar czvf $save_dir/$output_data.tmp data
# Move tmp to out.
mv $save_dir/$output_data.tmp $save_dir/$output_data