PY=../py/run.py
MODEL_DIR=../model/model_translate
TRAIN_PATH_FROM=../data/enfr/train_english.txt
DEV_PATH_FROM=../data/enfr/dev_english.txt
TEST_PATH_FROM=../data/enfr/test_english.txt
TRAIN_PATH_TO=../data/enfr/train_french.txt
DEV_PATH_TO=../data/enfr/dev_french.txt
TEST_PATH_TO=../data/enfr/test_french.txt


export CUDA_VISIBLE_DEVICES=1


python $PY --mode TRAIN --model_dir $MODEL_DIR \
    --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM \
    --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO \
    --batch_size 64 --from_vocab_size 10000 --to_vocab_size 10000 --size 128 --num_layers 2 \
    --n_epoch 10 --saveCheckpoint True --attention True --learning_rate 0.1 --keep_prob 0.7
    
