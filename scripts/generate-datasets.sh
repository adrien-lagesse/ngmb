# nodes=(50 100 200 500)
# densities=(2 4 8 16 24 32)
noises=(0.01 0.02 0.04 0.06 0.08 0.12 0.15 0.18 0.24 0.3 0.35)

for noise in ${noises[@]}; do
    echo "Generating dataset ER[100,12,$noise]"

    rye run gm-generate-er\
        -o "/scratch/jlagesse/ngmb-data/ER[100,12,$noise]" \
        --n-graphs 15000 \
        --n-val-graphs 1000 \
        --order 100 \
        --density 12 \
        --noise $noise 
done
