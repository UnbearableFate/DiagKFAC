import trainning_kit.common as common
#from trainning_kit.trainer import Trainer
from FNGD.fngd import FngdTrainer
args = common.merged_args_parser()
mgr = FngdTrainer(args)
mgr.train_and_evaluate()