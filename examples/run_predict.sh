
python3 -m layoutgcn.predictor \
    --data-dir ./data/ \
    --model-path ./outputs/final_model/ \
    --output-dir ./outputs/ \
    --batch-size 64 \
    --do-evaluate True
