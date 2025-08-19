import datetime
import trainning_kit.common as common
from trainning_kit.trainer import Trainer
args = common.merged_args_parser()
timeout = datetime.timedelta(seconds=20)
mgr = Trainer(args)
mgr.train_and_evaluate()