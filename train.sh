nvidia-smi
python train_net.py --config-file configs/swinb_srnet.yaml SOLVER.IMS_PER_GPU 4 TEST.EVAL_PERIOD 5000 OUTPUT_DIR g4090