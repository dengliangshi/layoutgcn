
export WANDB_MODE=offline
export WANDB_DIR=./outputs/

python3 -m layoutgcn.trainer \
    --run-name layoutgcn-wo-image \
    --data-dir ./data/ \
    --task-type information_extraction \
    --max-seq-length 45 \
    --max-num-nodes 32 \
    --do-train \
    --do-eval \
    --output-dir ./outputs/ \
    --logging-steps 10 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 8 \
    --learning-rate 0.001 \
    --save-strategy steps \
    --save-steps 500 \
    --eval-strategy steps \
    --eval-steps 100 \
    --save-total-limit 3 
