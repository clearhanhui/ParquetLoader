from dist_parquet_loader import DistParquetDataset,  DistParquetDataLoader

parquet_path = 'synthetic_data'
dataset = DistParquetDataset(parquet_path)
dataloader = DistParquetDataLoader(dataset, batch_size=9, shuffle=False, num_workers=0)


for i, batch in enumerate(dataloader):
    print(i, batch)
