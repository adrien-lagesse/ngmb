nvidia-smi

rye run gm-train \
    --dataset  "/scratch/jlagesse/ngmb-data/PCQM4Mv2[0.04]" \
    --experiment "PCQM4Mv2-PE" \
    --run-name "GatedGCN-small PCQM4Mv2[0.04]" \
    --epochs 500 \
    --batch-size 100 \
    --cuda \
    --log-frequency 25 \
    --profile \
    --model GatedGCN \
        --layers 4\
        --features 48 \
        --out-features 32 \
    --optimizer adam-one-cycle \
        --max-lr 3e-3 \
        --start-factor 5 \
        --end-factor 500 \
        --grad-clip 0.1