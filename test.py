import torch.distributed as dist
import torch.multiprocessing as mp
from parquet_loader import DistParquetDataset,  DistParquetDataLoader
parquet_path = 'synthetic_data'
world_size = 2

def run(rank):
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:23456',
        rank=rank,
        world_size=world_size,
    )
    dataset = DistParquetDataset(parquet_path)
    dataloader = DistParquetDataLoader(dataset, batch_size=66, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        if rank == 0:
            print(rank, i, batch.shape)
        dist.barrier() 


if __name__ == '__main__':
    mp.start_processes(run, nprocs=world_size, start_method='spawn')
