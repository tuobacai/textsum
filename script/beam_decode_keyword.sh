PY=../python/predict.py
BLEU=../python/multi-bleu.perl
MODEL_DIR=../model/model_keyword
TRAIN_PATH_FROM=../data/keyword/train_content.txt
DEV_PATH_FROM=../data/keyword/dev_content.txt
TEST_PATH_FROM=../data/keyword/test_content.txt
TRAIN_PATH_TO=../data/keyword/train_title.txt
DEV_PATH_TO=../data/keyword/dev_title.txt
TEST_PATH_TO=../data/keyword/test_title.txt
DECODE_OUTPUT=../data/keyword/test.output

export CUDA_VISIBLE_DEVICES=1


python $PY --mode BEAM_DECODE --model_dir $MODEL_DIR \
    --test_path_from $TEST_PATH_FROM \
    --beam_size 10 --from_vocab_size 10000 --to_vocab_size 10000 --size 128 --num_layers 2 \
    --attention False --print_beam True --decode_output $DECODE_OUTPUT

