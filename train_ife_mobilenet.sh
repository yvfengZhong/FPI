export CUDA_VISIBLE_DEVICES="2" 

python scripts/main.py --new_cfg "data.name='lasc2'" "data.crop='False'" "data.single='True'" \
        "train.network='mobilenet'" "train.criterion='L2'" "train.epochs=100" "train.batch_size=8" "solver.learning_rate=0.00005" \
