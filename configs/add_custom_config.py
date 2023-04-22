from detectron2.config import CfgNode as CN

def add_custom_config(cfg):
    cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""
    cfg.MODEL.BACKBONE.NUM_FEATURES = []

    cfg.MODEL.FPN = CN()
    cfg.MODEL.FPN.DIM = 256
    
    cfg.MODEL.IOR_MASK_ENCODER = CN()
    cfg.MODEL.IOR_MASK_ENCODER.HIDDEN_DIM = 32
    cfg.MODEL.IOR_MASK_ENCODER.NUM_HEAD = 8

    cfg.MODEL.IOR_TRANSFORMER = CN()
    cfg.MODEL.IOR_TRANSFORMER.DIM = 256
    cfg.MODEL.IOR_TRANSFORMER.FFN_DIM = 512
    cfg.MODEL.IOR_TRANSFORMER.FFN_DROP = 0.0

    cfg.MODEL.IORHEAD = CN()
    cfg.MODEL.IORHEAD.DIM = 256
    cfg.MODEL.IORHEAD.HIDDEN_DIM = 512
    cfg.MODEL.IORHEAD.OUT_DIM = 256

    cfg.MODEL.IOR_DECODER_BLOCK = CN()
    cfg.MODEL.IOR_DECODER_BLOCK.USED_LEVELS = (-1,-2,-3)

    cfg.MODEL.IOR_DECODER = CN()
    cfg.MODEL.IOR_DECODER.NUM_BLOCKS = 2
    cfg.MODEL.IOR_DECODER.LOSS_WEIGHTS = [1.0, 1.0]

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

    cfg.DATASETS.ROOT = ""
    
    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False

    cfg.INPUT.FT_SIZE_TRAIN = 800
    cfg.INPUT.FT_SIZE_TEST = 800