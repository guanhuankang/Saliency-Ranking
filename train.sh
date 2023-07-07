nvidia-smi
python train_net.py --config-file configs/base_resnext50.yaml SOLVER.IMS_PER_GPU 1 SOLVER.STEPS "(50,80)" SOLVER.MAX_ITER 100 OUTPUT_DIR output/debug SOLVER.BASE_LR 0.00001 DATALOADER.NUM_WORKERS 1
