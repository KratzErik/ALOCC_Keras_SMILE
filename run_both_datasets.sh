#!/bin/bash

exp_name_prosivic=$1
exp_name_dreyeve=$2

python3 models.py -d prosivic -e 1000 -x $exp_name_prosivic &
python3 models.py -d dreyeve -e 1000 -x $exp_name_dreyeve
