nvidia-smi
python train_net.py --config-file configs/swinb_srnet.yaml MODEL.META_ARCHITECTURE "SRDynamic" SOLVER.IMS_PER_GPU 1 SOLVER.STEPS "(50,80)" SOLVER.MAX_ITER 100 OUTPUT_DIR output/htgc3t