export CUDA_VISIBLE_DEVICES="2" 

python scripts/test.py --new_cfg "data.name='lasc2'" "data.crop='False'" "data.single='True'" "model.dual='False'"  "train.network='vgg11'" \
    "train.pretrained='False'"