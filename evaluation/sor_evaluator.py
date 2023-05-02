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
            gt_ranks = [(r, m) for r,m in zip(inp["ranks"], inp["masks"])]
            gt_ranks.sort(key=lambda x: x[0], reverse=True)
            gts = [x[1] for x in gt_ranks]
            self.results.append(self.metrics.process(preds=out["masks"], gts=gts, thres=thres))
            self.image_names.append(inp["image_name"])

    def evaluate(self):
        return self.metrics.aggregate(self.results)