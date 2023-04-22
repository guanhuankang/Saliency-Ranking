python train_net.py --config-file configs/ior_sor_base.yaml --num-gpus 1 \
 DATASETS.ROOT /home/ti3/hkguan/saliencyranking/dataset/coco_sor/coco_sor \
 SOLVER.IMS_PER_BATCH 4 \
 INPUT.FT_SIZE_TRAIN 224