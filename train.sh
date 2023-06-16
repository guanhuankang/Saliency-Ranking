nvidia-smi
python train_net.py --config-file configs/swinb_srnet.yaml SOLVER.IMS_PER_GPU 1 TEST.EVAL_PERIOD 50 OUTPUT_DIR output/htgc3
