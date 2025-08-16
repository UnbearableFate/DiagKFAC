import datetime
import os
import trainning_kit.common as common
from trainning_kit.trainer import Trainer
import torch.distributed as dist
args = common.merged_args_parser()
timeout = datetime.timedelta(seconds=40)
ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    from mpi4py import MPI
    ompi_world_rank = MPI.COMM_WORLD.Get_rank()
    ompi_world_size = MPI.COMM_WORLD.Get_size()

print(f"OMPI World Size: {ompi_world_size}, OMPI World Rank: {ompi_world_rank} hostname {os.uname().nodename}")

dist.init_process_group(backend='nccl', init_method=f"file://{args.workspace_path}/share_files/pg_share{args.timestamp}", rank=ompi_world_rank,
                            world_size=ompi_world_size, timeout=timeout)
print("Distributed process group initialized.")
print(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")
if dist.get_rank() == 0:
    import logging
    logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(levelname)s - %(message)s')

mgr = Trainer(args)
dist.barrier()  # Ensure all processes are synchronized before starting training
mgr.train_and_evaluate()
dist.destroy_process_group()  # Clean up the process group after training


