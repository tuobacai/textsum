PY=../py/run.py
BLEU=../py/multi-bleu.perl
MODEL_DIR=../model/model_translate
TRAIN_PATH_FROM=../data/enfr/train_english.txt
DEV_PATH_FROM=../data/enfr/dev_english.txt
TEST_PATH_FROM=../data/enfr/test_english.txt
TRAIN_PATH_TO=../data/enfr/train_french.txt
DEV_PATH_TO=../data/enfr/dev_french.txt
TEST_PATH_TO=../data/enfr/test_french.txt
DECODE_OUTPUT=../data/enfr/test.output

export CUDA_VISIBLE_DEVICES=1


python $PY --mode BEAM_DECODE --model_dir $MODEL_DIR \
    --test_path_from $TEST_PATH_FROM \
    --beam_size 10 --from_vocab_size 10000 --to_vocab_size 10000 --size 128 --num_layers 2 \
    --attention True --print_beam True --decode_output $DECODE_OUTPUT

