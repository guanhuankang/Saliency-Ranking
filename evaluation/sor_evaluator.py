import numpy as np
from PIL import Image
import os

from detectron2.evaluation import DatasetEvaluator
from .metrics import Metrics

class SOREvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.metrics = Metrics(cfg.TEST.METRICS_OF_INTEREST)
        self.results = []
        self.image_names = []

    def reset(self):
        self.image_names = []
        self.results = []
        self.metrics.from_config(cfg=self.cfg)

    def process(self, inputs, outputs):
        thres = self.cfg.TEST.THRESHOLD
        for inp, out in zip(inputs, outputs):
            image_name = inp["image_name"]
            
            ## EVAL
            self.results.append(self.metrics.process(preds=out["masks"], gts=list(inp["masks"]), thres=thres))
            self.image_names.append(image_name)
            
            ## SAVE
            if self.cfg.EVAL_SAVE:
                out_path = os.path.join(self.cfg.OUTPUT_EVAL, self.dataset_name)
                os.makedirs(out_path, exist_ok=True)
                preds = [x.cpu().detach().numpy() for x in out["masks"]]
                n = len(preds)
                for i in range(n):
                    Image.fromarray((preds[i]*255).astype(np.uint8)).save(
                        os.path.join(out_path, f"{image_name}_{n-i}.png")
                    )
    
    def evaluate(self):
        return self.metrics.aggregate(self.results)