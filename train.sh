nvidia-smi
python train_net.py --config-file configs/swinb_srnet.yaml SOLVER.IMS_PER_GPU 2 TEST.EVAL_PERIOD 5000 OUTPUT_DIR output/main