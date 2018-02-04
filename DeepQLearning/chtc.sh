#!/bin/bash

# Simulate CHTC nested folders, the save directory.
mkdir CHTC_start

# Move input to save directory, CHTC_start.
mv *.tar.gz CHTC_start
cd CHTC_start

# Run DQN.
sh /DQN/dqn.sh /DQN/ $PWD/ "--chtc=True --t_learn_start=10"