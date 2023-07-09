nvidia-smi
python train_net.py --config-file configs/multiq_swinb.yaml SOLVER.IMS_PER_GPU 1 SOLVER.STEPS "(500,800)" SOLVER.MAX_ITER 1000 OUTPUT_DIR output/debug SOLVER.BASE_LR 0.00001 DATALOADER.NUM_WORKERS 1
