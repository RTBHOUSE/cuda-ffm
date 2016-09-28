#!/bin/bash

set -exo pipefail

rm -rf tmp
mkdir -p tmp

# text_to_bin test

./bin/text_to_bin < examples/dataset1.txt > tmp/dataset1.bin
./bin/text_to_bin < examples/dataset2.txt > tmp/dataset2.bin

# bin_to_text test

./bin/bin_to_text tmp/dataset1.bin > tmp/dataset1.txt
./bin/bin_to_text tmp/dataset2.bin > tmp/dataset2.txt

diff examples/dataset1.txt tmp/dataset1.txt
diff examples/dataset2.txt tmp/dataset2.txt

# shuffler test

./bin/shuffler tmp/dataset1.bin tmp/dataset2.bin > tmp/shuffled.bin

bin/bin_to_text tmp/shuffled.bin > tmp/shuffled.txt
num_samples=`cat tmp/shuffled.txt | wc -l`
test "$num_samples" = 30000 || exit 1
num_ds1_chunks=`diff tmp/dataset1.txt tmp/shuffled.txt |grep -v '>' |wc -l || true`
test "$num_ds1_chunks" -gt 5000 || exit 1

# splitter test

./bin/splitter tmp/shuffled.bin 0.10 tmp/testing.bin tmp/training.bin

bin/bin_to_text tmp/testing.bin > tmp/testing.txt
bin/bin_to_text tmp/training.bin > tmp/training.txt

num_samples=`wc -l tmp/testing.txt tmp/training.txt | grep total | awk '{print $1}'`
test "$num_samples" = 30000 || exit 1

num_samples=`cat tmp/testing.txt | wc -l`
test "$num_samples" -lt 3300 || exit 1
test "$num_samples" -gt 2700 || exit 1

diff <(sort tmp/shuffled.txt) <(sort tmp/testing.txt tmp/training.txt)

# test learning

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

stdbuf -o0 ./bin/trainer --testingDatasetPath   tmp/testing.bin         \
              --trainingDatasetPath  tmp/training.bin                   \
              --maxBatchSize         200                                \
              --l2reg                0.000001                           \
              --maxNumEpochs         50                                 \
              --learningRate         0.07                               \
              --samplingFactor       100                                \
              --outputModelFilePath  tmp/model.out | stdbuf -i0 -o0 tee tmp/learn.log

num_weights=`cat tmp/model.out | wc -l`
test "$num_weights" = $((1000000*27*4+1)) || exit 1

# test prediction

learn_log_loss=`tail -n4 tmp/learn.log | grep 'Best model log-loss' | awk '{print $4}'`
best_predict_log_loss=`./bin/predict --textModelPath tmp/model.out --datasetPath tmp/testing.bin --samplingFactor 100.0 | tail -n1 | awk '{print $2}'`
last_predict_log_loss=`./bin/predict --textModelPath tmp/model.out.last --datasetPath tmp/testing.bin --samplingFactor 100.0 | tail -n1 | awk '{print $2}'`

echo $learn_log_loss $best_predict_log_loss $last_predict_log_loss
echo "assert abs($learn_log_loss - $best_predict_log_loss) < 0.000001" | python
echo "assert $last_predict_log_loss >= $best_predict_log_loss" | python

# test learning (2)

./bin/trainer --testingDatasetPath   tmp/dataset1.bin        \
              --trainingDatasetPath  tmp/dataset2.bin        \
              --maxBatchSize         200                     \
              --l2reg                0.000001                \
              --maxNumEpochs         8                       \
              --learningRate         0.07                    \
              --samplingFactor       100                     \
              --seed                 456                     \
              --outputModelFilePath  tmp/model.out | tee tmp/learn.log

learn_log_loss=`tail -n4 tmp/learn.log | grep 'Best model log-loss' | awk '{print $4}'`
echo "assert abs($learn_log_loss - 0.0337498) < 0.000001" | python

echo SUCCESS
