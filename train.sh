#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:4             # Using 1 GPU card
#SBATCH --mem=256GB               # Request 50GB memory
#SBATCH --time=0-11:59:00        # Time limit day-hrs:min:sec
#SBATCH --output=output/output.log   # Standard output
#SBATCH --error=output/error.err    # Standard error log

nvidia-smi
python train_net.py --config-file configs/ior_sor_detr.yaml --num-gpus 4 SOLVER.IMS_PER_BATCH 16 INPUT.FT_SIZE_TRAIN 1024 INPUT.FT_SIZE_TEST 1024 TEST.EVAL_PERIOD 5000 DATASETS.ROOT /home/huankguan2/saliencyranking/dataset/coco_sor SOLVER.BASE_LR 0.0001 SOLVER.STEPS [27000,28500] SOLVER.MAX_ITER 30000