nvidia-smi
python train_net.py --config-file configs/swinb_qinst.yaml SOLVER.IMS_PER_GPU 1 SOLVER.STEPS "(50,80)" SOLVER.MAX_ITER 100 MODEL.META_ARCHITECTURE "SRParallel" OUTPUT_DIR output/htgc3t
