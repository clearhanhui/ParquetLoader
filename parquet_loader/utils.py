import os
from typing import List, Tuple, Dict

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import get_worker_info
from dataclasses import dataclass, field


@dataclass
class RowGroupInterval:
    file_index: int = 0
    row_group_index: int = 0
    global_row_start: int = 0
    global_row_end: int = 0
    local_row_start: int = 0
    local_row_end: int = 0


@dataclass
class ParquetMetadata:
    file_path: str
    num_rows: int = 0
    num_row_groups: int = 0
    num_rows_per_row_group: list = field(default_factory=list)


def associate_to_workers(
    metas: List[ParquetMetadata],
    world_size: int = 1,
    num_workers: int = 1,
    current_rank: int = 0,
    current_worker_rank: int = 0,
    drop_last: bool = False,
    batch_size: int = 1,
) -> Tuple[Dict[int, List[RowGroupInterval]], int]:  
    
    total_num_rows = sum([meta.num_rows for meta in metas])
    num_rows_per_ranks = [
        total_num_rows // world_size + total_num_rows % world_size # TODO: check its correctness
        if rank == world_size - 1 and not drop_last
        else total_num_rows // world_size
        for rank in range(world_size)
    ]
    if drop_last:
        ratio = num_workers * batch_size
        num_rows_per_ranks = [ratio * int(item // ratio) for item in num_rows_per_ranks]
    num_rows_per_workers = np.array([
        [   
            num_rows_per_ranks[rank] // num_workers + num_rows_per_ranks[rank] % num_workers
            if worker_rank == 0 and not drop_last # worker 0 gets the remainder
            else num_rows_per_ranks[rank] // num_workers
            for worker_rank in range(num_workers)
        ]
        for rank in range(world_size)
    ])

    row_group_intervals = []
    global_rows = 0
    for i, meta in enumerate(metas):
        intervals_per_file = []
        for j, local_rows in enumerate(meta.num_rows_per_row_group):
            intervals_per_file.append(
                RowGroupInterval(i, j, global_rows, global_rows+local_rows, 0, local_rows)
            )
            global_rows += local_rows
        row_group_intervals.append(intervals_per_file)


    start_row_index_per_workers = [0] + np.cumsum(num_rows_per_workers).tolist()
    current_global_row_start = start_row_index_per_workers[current_rank * num_workers + current_worker_rank]
    current_global_row_end = start_row_index_per_workers[current_rank * num_workers + current_worker_rank + 1]
    current_intervals = {}
    for intervals_per_file in row_group_intervals:
        for itv in intervals_per_file:
            if itv.global_row_end <= current_global_row_start:
                continue
            if itv.global_row_start >= current_global_row_end:
                break
            
            if current_global_row_start > itv.global_row_start and \
               current_global_row_start < itv.global_row_end:
                itv.local_row_start = current_global_row_start - itv.global_row_start
                itv.global_row_start = current_global_row_start
            
            if current_global_row_end > itv.global_row_start and \
               current_global_row_end < itv.global_row_end:
                itv.local_row_end = current_global_row_end - itv.global_row_start
                itv.global_row_end = current_global_row_end

            if itv.file_index not in current_intervals:
                current_intervals[itv.file_index] = [itv]
            else:
                current_intervals[itv.file_index].append(itv)
                                                 
    
    return current_intervals, current_global_row_end-current_global_row_start


def detect_distributed_env():
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        # Note: On multi node CPU, the number of nodes won't be correct.
        if torch.cuda.is_available() and world_size // torch.cuda.device_count() >= 1:
            num_nodes = world_size // torch.cuda.device_count()
        else:
            num_nodes = 1

        # If you are using multiple nodes, we assume you are using all the GPUs.
        # On single node, a user can be using only a few GPUs of the node.
        if torch.cuda.is_available() and num_nodes > 1 and world_size % torch.cuda.device_count() != 0:
            raise RuntimeError("The world size should be divisible by the number of GPUs.")
    else:
        world_size = None
        global_rank = 0
        num_nodes = 1

    if world_size is None or world_size == -1:
        world_size = 1

    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    global_rank = int(os.environ.get("GLOBAL_RANK", global_rank))
    num_nodes = int(os.environ.get("NNODES", num_nodes))

    return world_size, global_rank, num_nodes


def detect_worker_env():
    """Automatically detects the number of workers and the current rank.

    Note:
        This only works reliably within a dataloader worker as otherwise the necessary information won't be present.
        In such a case it will default to 1 worker

    """
    worker_info = get_worker_info()
    num_workers = worker_info.num_workers if worker_info is not None else 1
    current_worker_rank = worker_info.id if worker_info is not None else 0

    return num_workers, current_worker_rank


