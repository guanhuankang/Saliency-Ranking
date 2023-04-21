from detectron2.config import CfgNode as CN

def add_custom_config(cfg):
    cfg.MODEL.WEIGHTS = ""
    
    cfg.DATASETS.ROOT = ""
    
    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False