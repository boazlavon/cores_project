export COREF_DIR="/home/yandex/AMNLP2021/boazlavon/coref_repos/s2e-coref"
export OUTPUT_DIR="$COREF_DIR/output_$(date +%s)"
export CACHE_DIR="$COREF_DIR/cache"
export MODEL_DIR="$COREF_DIR/model"
export DATA_DIR="$COREF_DIR/coref_data"
export SPLIT_FOR_EVAL=test

python run_coref.py \
        --output_dir=$OUTPUT_DIR \
        --cache_dir=$CACHE_DIR \
        --model_type=longformer \
        --model_name_or_path=$MODEL_DIR \
        --tokenizer_name=allenai/longformer-large-4096 \
        --config_name=allenai/longformer-large-4096  \
        --train_file=$DATA_DIR/train.english.jsonlines \
        --predict_file=$DATA_DIR/test.english.jsonlines \
        --do_eval \
        --num_train_epochs=129 \
        --logging_steps=500 \
        --save_steps=3000 \
        --eval_steps=1000 \
        --max_seq_length=4096 \
        --train_file_cache=$DATA_DIR/train.english.4096.pkl \
        --predict_file_cache=$DATA_DIR/test.english.4096.pkl \
        --amp \
        --normalise_loss \
        --max_total_seq_len=5000 \
        --experiment_name=eval_model \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$OUTPUT_DIR/tb \
        --conll_path_for_eval=$DATA_DIR/$SPLIT_FOR_EVAL

echo $OUTPUT_DIR
