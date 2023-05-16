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

    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.EMBED_DIM = 256
    cfg.MODEL.DECODER.NUM_HEADS = 8
    cfg.MODEL.DECODER.HIDDEN_DIM = 1024
    cfg.MODEL.DECODER.DROPOUT_ATTN = 0.0
    cfg.MODEL.DECODER.DROPOUT_FFN = 0.0
    cfg.MODEL.DECODER.NUM_QUERIES = 100
    cfg.MODEL.DECODER.NUM_BLOCKS = 3
    cfg.MODEL.DECODER.FEATURE_KEYS = ["res5", "res4", "res3"]

    cfg.MODEL.HEAD = CN()
    cfg.MODEL.HEAD.EMBED_DIM = 256
    cfg.MODEL.HEAD.NUM_HEADS = 8
    cfg.MODEL.HEAD.HIDDEN_DIM = 1024
    cfg.MODEL.HEAD.DROPOUT_ATTN = 0.0
    cfg.MODEL.HEAD.DROPOUT_FFN = 0.0
    cfg.MODEL.HEAD.NUM_BLOCKS = 2

    cfg.DATASETS.ROOT = ""

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