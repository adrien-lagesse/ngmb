# nodes=(50 100 200 500)
# densities=(2 4 8 16 24 32)
noises=(0.01 0.02 0.04 0.06 0.08 0.12 0.18 0.24 0.3 0.35)

for noise in ${noises[@]}; do
    echo "Generating dataset pcqm4mv2[$noise]"

    rye run gm-generate-pcqm4mv2 \
        -o "/scratch/jlagesse/ngmb-data/PCQM4Mv2[$noise]" \
        --n-graphs 15000 \
        --n-val-graphs 500 \
        --noise $noise 
done
