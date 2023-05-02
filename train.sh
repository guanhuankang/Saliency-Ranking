python train_net.py --config-file configs/ior_sor_base.yaml --num-gpus 1 \
 DATASETS.ROOT D:\SaliencyRanking\dataset\coco_sor \
 SOLVER.IMS_PER_BATCH 4 \
 INPUT.FT_SIZE_TRAIN 224 \
 INPUT.FT_SIZE_TEST 224 \
 EVAL_PERIOD 10