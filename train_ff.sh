export CUDA_VISIBLE_DEVICES="2" 

python scripts/main.py --new_cfg "data.name='lasc2'" "data.crop='False'" "data.single='True'" \
        "model.dual='True'" "train.dual_checkpoint='./checkpoints/mlp'" "train.checkpoint='./checkpoints/resnet18'" \
        "model.add_attention='True'" "train.freeze='True'" \
        "train.network='resnet18'" "train.criterion='L2'" "train.epochs=100" "train.batch_size=8" "solver.learning_rate=0.0005" \