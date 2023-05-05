nvidia-smi
python train_net.py --config-file configs/ior_sor_detr.yaml --num-gpus 1 \
SOLVER.IMS_PER_BATCH 4 INPUT.FT_SIZE_TRAIN 1024 INPUT.FT_SIZE_TEST 1024 \
TEST.EVAL_PERIOD 5000 \
DATASETS.ROOT /home/huankguan2/saliencyranking/dataset/coco_sor \
SOLVER.BASE_LR 0.0001 \
SOLVER.STEPS [27000,28500]\
SOLVER.MAX_ITER 30000