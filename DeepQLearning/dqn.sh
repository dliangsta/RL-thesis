#!/bin/bash 
run_dir=${1:-"./"}
save_dir=${2:-"./"}
extra_args=${3:-""}
echo "Docker container: Running DQN in directory $run_dir from $save_dir"
# CHTC run.
# python3 -u $1"main.py" --run_dir=$run_dir --save_dir=$save_dir --is_train=False
# CHTC train.
if [ -f output_data.tar.gz ]; then
    mv output_data.tar.gz input_data.tar.gz
    echo "Docker container: Output data overwriting input data."
else
    echo "Docker container: No output data to overwrite input data with."
fi

rm -rf data
tar xzvf input_data.tar.gz -C $save_dir

echo "\nDocker container: ls -al"
ls -al
echo "\nDocker container: pwd"
pwd
echo "\n"

python3 -u $1"main.py" --run_dir=$run_dir --save_dir=$save_dir $extra_args
echo "Docker container: Done."