#!/bin/bash
#for i in 0 1 2 3

while true; do
  python train.py --device 0 --batch-size 16 --img 416 416 --data dota.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name Results --epochs 500

done

# 36:34 2080ti
# 21:58 V100
# 63:00 T4