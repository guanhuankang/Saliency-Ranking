#!/bin/bash
#SBATCH --partition=special_cs
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=256GB               # Request 50GB memory
#SBATCH --time=10-11:59:00        # Time limit day-hrs:min:sec
#SBATCH --output=log/output.log   # Standard output
#SBATCH --error=log/error.err    # Standard error log

date
echo PID:$$
cd /home/grads/huankguan2/projects/saliencyranking/codebase/apps/instance_seg
pwd
nvidia-smi

output=output/instance-seg-bs16
python train_net.py --config-file configs/ior_sor_detr.yaml SOLVER.IMS_PER_BATCH 16 INPUT.FT_SIZE_TRAIN 1024 INPUT.FT_SIZE_TEST 1024 TEST.EVAL_PERIOD 5000 DATASETS.ROOT /home/huankguan2/saliencyranking/dataset/coco_sor MODEL.WEIGHTS /home/huankguan2/saliencyranking/codebase/pretrained/swin_base_patch4_window12_384_22k.pth SOLVER.BASE_LR 0.0001 SOLVER.STEPS [180000,190000] SOLVER.MAX_ITER 200000 OUTPUT_DIR $output
echo done