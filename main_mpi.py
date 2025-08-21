import datetime
import os
import torch.distributed as dist
from trainning_kit.yaml_config_parser import merged_args_parser
from trainning_kit.trainer import Trainer

args = merged_args_parser()
ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    from mpi4py import MPI
    ompi_world_rank = MPI.COMM_WORLD.Get_rank()
    ompi_world_size = MPI.COMM_WORLD.Get_size()

print(f"OMPI World Size: {ompi_world_size}, OMPI World Rank: {ompi_world_rank} hostname {os.uname().nodename}")
timeout = datetime.timedelta(seconds=30)
dist.init_process_group(backend='nccl', timeout=timeout, rank=ompi_world_rank, world_size=ompi_world_size)
print("Distributed process group initialized.")
print(f"World Size: {dist.get_world_size()}, World Rank: {dist.get_rank()} hostname {os.uname().nodename}")
if dist.get_rank() == 0:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trainer = Trainer(args)
dist.barrier()  # 确保所有进程在开始前同步
trainer.train_and_evaluate()
dist.destroy_process_group()
trainer.print_after_train()
exit(0)
