from detectron2.engine import (
    DefaultTrainer, 
    default_argument_parser, 
    default_setup, 
    launch
)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm, logger
from detectron2.evaluation import (
    verify_results
)

from configs.add_custom_config import add_custom_config
from dataset import register_sor_dataset
from Toy import ToyModel

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    logger.setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="toy")
    return cfg

def main(args):
    cfg = setup(args)
    ## register sor dataset before starts training
    register_sor_dataset(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model=model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )