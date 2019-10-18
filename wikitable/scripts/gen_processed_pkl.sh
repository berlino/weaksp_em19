DATA_DIR="../dataset/WikiTable"
TRAIN_FILE=$DATA_DIR"/data_split_1/train_split.jsonl"
DEV_FILE=$DATA_DIR"/data_split_1/dev_split.jsonl"
TEST_FILE=$DATA_DIR"/test_split.jsonl"
TABLE_FILE=$DATA_DIR"/tables.jsonl"
EMBED_FILE="../glove/glove.42B.300d.txt"
OUTPUT_FILENAME="processed/wikitable_glove_42B_minfreq_3.pkl"

python reader/reader.py \
    -train_file=$TRAIN_FILE \
    -dev_file=$DEV_FILE \
    -test_file=$TEST_FILE \
    -table_file=$TABLE_FILE \
    -embed_file=$EMBED_FILE \
    -output_file=$OUTPUT_FILENAME


