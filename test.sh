nvidia-smi
python train_net.py --config-file configs/swinb_srnet.yaml --eval-only DATASETS.TEST "('irsr_test',)" MODEL.WEIGHTS $1 OUTPUT_DIR output/full_dropout TEST.EVAL_SAVE False
