import datetime
import trainning_kit.common as common
from trainning_kit.trainer import Trainer
import logging
logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(levelname)s - %(message)s')
args = common.merged_args_parser()
timeout = datetime.timedelta(seconds=20)
mgr = Trainer(args)
mgr.train_and_evaluate()