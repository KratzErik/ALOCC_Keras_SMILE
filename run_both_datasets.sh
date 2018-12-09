#!/bin/bash

exp_name_prosivic=$1
exp_name_dreyeve=$2

python3 models.py -d prosivic -e 500 -x $exp_name_prosivic &
python3 models.py -d dreyeve -e 500 -x $exp_name_dreyeve
