from trainning_kit.yaml_config_parser import merged_args_parser
from trainning_kit.trainer import Trainer
args = merged_args_parser()
trainer = Trainer(args)
trainer.train_and_evaluate()
trainer.print_after_train()
exit(0)