export CUDA_VISIBLE_DEVICES="2" 

python scripts/test.py --new_cfg "data.name='lasc2'" "data.crop='False'" "data.single='True'" \
        "model.dual='True'" "train.finetune='./checkpoints/attention'" \
        "model.add_attention='True'" "train.freeze='True'" \
        "train.network='resnet18'" \