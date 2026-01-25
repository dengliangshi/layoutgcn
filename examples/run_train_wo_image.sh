
export WANDB_MODE=offline
export WANDB_DIR=./outputs/


python3 -m layoutgcn.preprocess \
    --data-dir ./data/ \
    --output-dir ./outputs/ \
    --max-seq-length 45 \
    --max-num-nodes 32 \
    --task-type information_extraction


python3 -m layoutgcn.trainer \
    --run-name layoutgcn-wo-image \
    --data-dir ./outputs/data/ \
    --task-type information_extraction \
    --max-seq-length 45 \
    --max-num-nodes 32 \
    --use-crf True \
    --do-train \
    --do-eval \
    --output-dir ./outputs/ \
    --logging-steps 10 \
    --num-train-epochs 100 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 8 \
    --learning-rate 0.001 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --save-strategy epoch \
    --eval-strategy epoch \
    --metric-for-best-model accuracy \
    --load-best-model-at-end True \
    --save-total-limit 3 
