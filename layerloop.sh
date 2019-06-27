#!/bin/bash
echo 5
for i in {1..17}
do
   python encoder.py -i ./aligned_realpics/64/chicago2.png -n layer$i --img_size 64 --layersF $i --steps 700 --loss 1*FACE --lr 0.05 --optimizer ADAM --mask_type SEGMENT
done