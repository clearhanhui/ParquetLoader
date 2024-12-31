import numpy as np
import pandas as pd
import pyarrow as pa

from table_io import TableIO

table = pa.Table.from_pandas(
    pd.DataFrame(np.random.randn(5000001, 100)))

tableio = TableIO()
tableio.write_table(table, '/tmp/test2025', file_size=500_000, chunk_size=100_000)
tableio.read_table('/tmp/test2025')
with tableio.iter_writter('/tmp/test2026') as writer:
    for batch_table in tableio.iter_batches('/tmp/test2025'):
        writer.write_batch(batch_table)
