#!/bin/bash

dataset=$1
exp_name=$2
epochs=$3

python3 models.py -d $dataset -x $exp_name -e $epochs &&
python3 test.py -d $dataset -x $exp_name -e final
