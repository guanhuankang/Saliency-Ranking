nvidia-smi
python train_net.py --config-file configs/mask2former_swinB.yaml MODEL.SIS_HEAD.NAME "Mask2Former" MODEL.GAZE_SHIFT_HEAD.NAME "GazeShift" SOLVER.IMS_PER_GPU 1 SOLVER.STEPS "(50,80)" SOLVER.MAX_ITER 100 OUTPUT_DIR output/debug SOLVER.BASE_LR 0.00001 DATALOADER.NUM_WORKERS 1
