from detectron2.config import CfgNode as CN

def add_custom_config(cfg, num_gpus=1):
    cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""
    cfg.MODEL.BACKBONE.NUM_FEATURES = [128, 256, 512, 1024]
    cfg.MODEL.BACKBONE.FEATURE_KEYS = ["res2", "res3", "res4", "res5"]

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.MODEL.NECK = CN()
    cfg.MODEL.NECK.DIM = 256
    cfg.MODEL.NECK.NUM_FEATURES = [128, 256, 512, 1024]
    cfg.MODEL.NECK.FEATURE_KEYS = ["res2", "res3", "res4", "res5"]

    cfg.MODEL.MODULES = CN()
    cfg.MODEL.MODULES.BBOX_DECODER = CN()
    cfg.MODEL.MODULES.BBOX_DECODER.NUM_BLOCKS = 2
    cfg.MODEL.MODULES.BBOX_DECODER.NUM_QUERIES = 100
    cfg.MODEL.MODULES.MASK_DECODER = CN()
    cfg.MODEL.MODULES.MASK_DECODER.NUM_BLOCKS = 2
    cfg.MODEL.MODULES.GAZE_SHIFT = CN()
    cfg.MODEL.MODULES.GAZE_SHIFT.NUM_BLOCKS = 2
    cfg.MODEL.MODULES.GAZE_SHIFT.SIGMA = 10.0
    cfg.MODEL.MODULES.GAZE_SHIFT.KERNEL_SIZE = 3
    cfg.MODEL.MODULES.FOVEAL = CN()
    cfg.MODEL.MODULES.FOVEAL.NUM_BLOCKS = 2
    cfg.MODEL.MODULES.FOVEAL.KEY_FEATURES = ["res5", "res4", "res3"]
    cfg.MODEL.MODULES.FOVEALQ = CN()
    cfg.MODEL.MODULES.FOVEALQ.NUM_BLOCKS = 4
    cfg.MODEL.MODULES.FOVEALQ.KEY_FEATURES = ["res5", "res4", "res3"]

    cfg.MODEL.COMMON = CN()
    cfg.MODEL.COMMON.EMBED_DIM = 256
    cfg.MODEL.COMMON.NUM_HEADS = 8
    cfg.MODEL.COMMON.HIDDEN_DIM = 1024
    cfg.MODEL.COMMON.DROPOUT_ATTN = 0.0
    cfg.MODEL.COMMON.DROPOUT_FFN = 0.0
    cfg.MODEL.COMMON.NUM_QUERIES = 100

    cfg.LOSS = CN()
    cfg.LOSS.WEIGHTS = CN()
    cfg.LOSS.WEIGHTS.OBJ_POS = 10.0
    cfg.LOSS.WEIGHTS.OBJ = 1.0
    cfg.LOSS.WEIGHTS.SAL = 1.0
    cfg.LOSS.WEIGHTS.MASK = 1.0
    cfg.LOSS.WEIGHTS.IOU = 1.0
    cfg.LOSS.WEIGHTS.SEL = 1.0

    cfg.DATASETS.ROOT = ""
    cfg.DATASETS.ENV = CN()
    cfg.DATASETS.ENV.WORK = ""
    cfg.DATASETS.ENV.GROUP4090 = ""
    cfg.DATASETS.ENV.BURGUNDY = ""
    cfg.DATASETS.ENV.HTGC = ""
    cfg.DATASETS.ENV.GROUP3090 = ""

    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.IMS_PER_GPU = 1
    cfg.SOLVER.ITERS_PER_STEP = 1
    cfg.SOLVER.NUM_GPUS = num_gpus
    cfg.SOLVER.TIMEOUT = 2

    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.UPPER_BOUND = False
    cfg.TEST.EVAL_SAVE = False
    cfg.TEST.METRICS_OF_INTEREST = ["mae"]
    cfg.TEST.THRESHOLD = 0.5

    cfg.INPUT.FT_SIZE_TRAIN = 800
    cfg.INPUT.FT_SIZE_TEST = 800