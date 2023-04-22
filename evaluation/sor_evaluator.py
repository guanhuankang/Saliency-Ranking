from detectron2.evaluation import DatasetEvaluator

class SOREvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name

    def reset(self):
        pass

    def process(self, inputs, outputs):
        return super().process(inputs, outputs)
    
    def evaluate(self):
        return super().evaluate()