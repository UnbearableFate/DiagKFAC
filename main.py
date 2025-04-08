import trainning_kit.common as common
from trainning_kit.trainer import Trainer

args = common.merged_args_parser()
mgr = Trainer(args)
mgr.train_and_evaluate()