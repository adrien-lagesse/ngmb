salloc --job-name=NGMB-Interact \
       --nodes=1 \
       --partition=gpu \
       --gres=gpu:1 \
       --nodelist=gpu[001,002,003,006,007,008,009,012,013] \
       --mem-per-gpu=20G \
       --cpus-per-gpu=8 \
       --time=47:00:00
