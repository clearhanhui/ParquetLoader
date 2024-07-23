from parquet_loader import DistParquetDataset,  DistParquetDataLoader

parquet_path = 'synthetic_data'
dataset = DistParquetDataset(parquet_path)
dataloader = DistParquetDataLoader(dataset, batch_size=66, shuffle=True, num_workers=2)


for i, batch in enumerate(dataloader):
    print(i, batch.shape)
