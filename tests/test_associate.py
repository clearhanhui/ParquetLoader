from parquet_loader.utils import ParquetMetadata, associate_to_workers

if __name__ == '__main__':
    test_metas = [
        ParquetMetadata(
            file_path = str(i),
            num_rows = 4,
            num_row_groups = 2,
            num_rows_per_row_group = [2 for _ in range(2)]
        ) for i in range(2)
    ]
    intervals, num_rows = associate_to_workers(
        test_metas, 
        world_size=1, 
        num_workers=1, 
        current_rank=0, 
        current_worker_rank=0,
        drop_last=False,
        batch_size=11,
    )
    print(num_rows)
    for itvs in intervals:
        for itv in itvs:
            print(itv)