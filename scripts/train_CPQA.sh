export MODEL_PATH="YOUR LLM PATH"
export SAVE_PATH="YOUR SAVE PATH"
export DATA_PATH="YOUR DATA PATH"
export WANDB_DISABLED=true
wandb offline

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 40 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 --rrhf_weight 0.1 > log.txt &
