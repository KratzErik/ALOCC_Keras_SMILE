#!/bin/bash

exp_name=$1
epochs=$2
epochs2=$3

python3 models.py -d prosivic -e $epochs -x $exp_name &&
python3 models.py -d dreyeve -e $epochs2 -x $exp_name
