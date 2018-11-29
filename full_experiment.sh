Â#!/bin/bash
exp_name=$1
epochs=$2

python3 models.py -x $exp_name -e $epochs &&
python3 test.py --exp_name $exp_name --load_epoch final
