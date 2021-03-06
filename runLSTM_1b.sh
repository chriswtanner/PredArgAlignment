#!/bin/sh
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/dc65/Documents/tools/cuda/lib64:/home/dc65/Documents/tools/cuda/extras/CUPTI/lib64"
# export CUDA_HOME=/home/dc65/Documents/tools/cuda

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/dc65/Documents/tools/cuda8.0/lib64"
export CUDA_HOME=/home/dc65/Documents/tools/cuda8.0

# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"
# export CUDA_HOME=/usr/local/cuda-8.0

cd /home/christanner/researchcode/PredArgAlignment/src/
echo $LD_LIBRARY_PATH
echo $CUDA_HOME

python Test.py $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15 $16 $17 $18 $19 $20 $21 $22 $23 $24 $25 $26
# python Test.py /data/people/christanner/ 42 false false cpu lstm 100 5 30 5 1.0
# $dn $sm $rc $dev lstm $hs$ns $ne$bs $lr
