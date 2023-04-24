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
            self.results.append(self.metrics.process(preds=out["masks"], gts=inp["masks"], thres=thres))
            self.image_names.append(inp["image_name"])

    def evaluate(self):
        return self.metrics.aggregate(self.results)