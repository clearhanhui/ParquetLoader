import os
import abc
import logging
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq 
import pyarrow.orc as orc
import pyarrow.csv as csv
import pyarrow.fs as fs
from fsspec.implementations.arrow import ArrowFSWrapper



PARQUET = 'parquet'
CSV = 'csv'
ORC = 'orc'


def resolve_filesystem(path):
    arrow_fs, _ = fs.FileSystem.from_uri(path)
    return ArrowFSWrapper(arrow_fs)

def get_file_format(file_path):
    format = file_path.split('.')[-1]
    assert format in [PARQUET, CSV, ORC], f'Unsupported file format: {format}'
    return format


def ceil_divide(a, b):
    return (a + b - 1) // b

def split_table(table, num_splits):
    num_rows_split, extras = divmod(table.num_rows, num_splits)
    split_sizes = extras * [num_rows_split+1] + (num_splits-extras) * [num_rows_split]

    start_idx = 0
    split_tables = []
    for split_size in split_sizes:
        split_tables.append(table.slice(start_idx, split_size))
        start_idx += split_size
    return split_tables


class TableReader(abc.ABC):
    _format = None

    def __init__(
        self,
        paths: List[str],
        columns: Optional[List[str]] = None,
        filesystem = None
    ) -> None:
        self.paths = paths
        self.columns = columns
        self.filesystem = filesystem

    def read_table(self):
        dataset = ds.dataset(self.paths, format=self._format, filesystem=self.filesystem)
        return dataset.to_table(columns=self.columns)

    def iter_batches(self, batch_size=1024):
        table_list = []
        left_table = None
        num_rows_need = batch_size
        for shard_table in self.iter_shard():
            table_list.append(shard_table)
            num_rows_need -= shard_table.num_rows
            if num_rows_need > 0:
                continue
            else:
                left_table = pa.concat_tables(table_list)
                while num_rows_need <= 0:
                    batch_table = left_table.slice(0, batch_size)
                    yield batch_table
                    left_table = left_table.slice(batch_size)
                    num_rows_need += batch_size
                table_list = [left_table]

        if len(table_list) > 0:
            left_table = pa.concat_tables(table_list)
            if left_table.num_rows == 0:
                return
            batch_table = left_table.slice(0, batch_size)
            yield batch_table
    
    def iter_shard(self):
        pass

    def __iter__(self):
        return self.iter_shard()

class ParquetReader(TableReader):
    _format = PARQUET

    def iter_shard(self,):
        """Iterate row groups"""
        for fpath in self.paths:
            f = pq.ParquetFile(
                source=fpath, 
                filesystem=self.filesystem,
                pre_buffer=True,
                memory_map=True,
            )
            for rg in range(f.num_row_groups):
                yield f.read_row_group(rg, columns=self.columns)
            f.close()

class ORCReader(TableReader):
    _format = ORC

    def iter_shard(self,):
        """Iterate stripes"""
        for fpath in self.paths:
            f = self.filesystem.open_input_file(fpath)
            orc_file = orc.ORCFile(f)
            for stripe in range(orc_file.nstripes):
                yield orc_file.read_stripe(stripe, columns=self.columns)
            f.close() 
                
class CSVReader(TableReader):
    _format = CSV

    def iter_shard(self):
        """Iterate files"""
        for fpath in self.paths:
            f = self.filesystem.open_input_file(fpath)
            yield csv.read_csv(f, read_options=csv.ReadOptions(column_names=self.columns))
            f.close()

class ReaderFactory:
    _SUPPORTED_FORMATS = {
        PARQUET: ParquetReader,
        ORC: ORCReader, 
        CSV: CSVReader
    }

    @classmethod
    def create_reader(
        cls, 
        format: str, 
        paths: List[str],
        columns: Optional[List[str]] = None,
        filesystem = None
    ) -> 'TableReader':
        assert cls.check_format(format), f"Unsupported format: {format}"
        reader_class = cls._SUPPORTED_FORMATS[format]
        return reader_class(paths, columns, filesystem)

    @classmethod
    def register(cls, format: str, reader_class: type):
        if cls.check_format(format):
            logging.warning(f"Format {format} already exists. Overwriting")
        cls._SUPPORTED_FORMATS[format.lower()] = reader_class

    @classmethod
    def check_format(self, format: str) -> bool:
        return format in self._SUPPORTED_FORMATS



class TableWriter(abc.ABC):
    _format = None

    def __init__(self, filesystem) -> None:
        self.filsystem = filesystem

    def write_file(self):
        pass 

    def write_table(
        self, 
        table: pa.Table, 
        table_dir: str,
        file_size: int = 100_000,
        chunk_size: int = 10_000, # row_group / stripe
    ) -> str:
        file_size = max(file_size, chunk_size)
        num_partitions = ceil_divide(table.num_rows, file_size)  
        max_workers = min(num_partitions, os.cpu_count() or 1)

        if max_workers == 1:
            file_path = os.path.join(table_dir, f'part-00000.{self._format}')
            success = self.write_file(table, file_path, chunk_size)
            return table_dir
        
        part_tables = split_table(table, num_partitions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.write_file, 
                    part_table, 
                    os.path.join(table_dir, f'part-{idx:05d}.{self._format}'), 
                    chunk_size, 
                ) 
                for idx, part_table in enumerate(part_tables)
            ]
            results = [future.result() for future in as_completed(futures)]

        if not all(results):
            raise RuntimeError("Some dataframe partitions failed to write. Check error logs")
        return table_dir


class ParquetWriter(TableWriter):
    _format = PARQUET
    
    def write_file(
        self, 
        table: pa.Table, 
        file_path: str,
        row_group_size: int = 10_000,
    ) -> bool:
        try:
            pq.write_table(table, file_path, row_group_size, filesystem=self.filsystem)
            return True
        except Exception as e:
            logging.error(f"Error writing table to {file_path}: {e}")
            return False
        
class ORCWriter(TableWriter):
    _format = ORC

    def write_file(
        self,
        table: pa.Table,
        file_path: str,
        stripe_size: int = 10_000,
    ) -> bool:
        try:
            orc.write_table(table, file_path, filesystem=self.filsystem, stripe_size=stripe_size)
            return True
        except Exception as e:
            logging.error(f"Error writing table to {file_path}: {e}")
            return False


class IterWriter:
    def __init__(self, writer: TableWriter, format: str):
        self.writer = writer
        self.format = format
        self.buffer_table = None 
        self.file_index = 0
    
    def config(self, table_dir, file_size=100_000, chunk_size=10_000):
        file_size = max(file_size, chunk_size)
        self.table_dir = table_dir 
        self.file_size = file_size
        self.chunk_size = chunk_size

    def write_batch(self, table: pa.Table):
        if self.buffer_table is None:
            self.buffer_table = table
        else:
            self.buffer_table = pa.concat_tables([self.buffer_table, table])
        self._flush()

    def _flush(self, flush_buffer=False):
        if self.buffer_table is None:
            return
        
        while self.buffer_table.num_rows > self.file_size:
            part_table = self.buffer_table.slice(0, self.file_size)
            part_path = os.path.join(self.table_dir, f'part-{self.file_index:05d}.{self.format}')
            self.writer.write_file(part_table, part_path, self.chunk_size)
            self.file_index += 1
            self.buffer_table = self.buffer_table.slice(self.file_size, None)

        if flush_buffer and self.buffer_table.num_rows > 0:
            part_path = os.path.join(self.table_dir, f'part-{self.file_index:05d}.{self.format}')
            self.writer.write_file(self.buffer_table, part_path, self.chunk_size)
            self.file_index += 1
            self.buffer_table = None

    def __enter__(self, *args, **kwargs):
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush(flush_buffer=True)


class WriterFactory:
    _SUPPORTED_FORMATS = {
        PARQUET: ParquetWriter,
        ORC: ORCWriter,
    }

    @classmethod
    def create_writer(cls, format: str, filesystem) -> 'TableWriter':
        assert cls.check_format(format), f"Unsupported format: {format}"
        writer_class = cls._SUPPORTED_FORMATS[format]
        return writer_class(filesystem)
    
    @classmethod
    def create_iter_writer(cls, format: str, filesystem) -> IterWriter:
        assert cls.check_format(format), f"Unsupported format: {format}"
        writer_class = cls._SUPPORTED_FORMATS[format]
        writer = writer_class(filesystem)
        return IterWriter(writer, format)
    
    @classmethod
    def register(cls, format: str, writer_class: type):
        if cls.check_format(format):
            logging.warning(f"Format {format} already exists. Overwriting")
        cls._SUPPORTED_FORMATS[format.lower()] = writer_class

    @classmethod
    def check_format(self, format: str) -> bool:
        return format in self._SUPPORTED_FORMATS


class TableIO:
    def read_table(
        self, 
        table_dir: str, 
        columns: Optional[List[str]] = None,
    ) -> pa.Table:
        logging.info(f"Reading table from: {table_dir}")
        filesystem = resolve_filesystem(table_dir)
        file_paths = filesystem.ls(table_dir)
        format = get_file_format(file_paths[0])
        reader = ReaderFactory.create_reader(format, file_paths, columns, filesystem.fs) 
        table = reader.read_table()
        return table


    def iter_batches(
        self,
        table_dir: str, 
        columns: Optional[List[str]] = None,
        batch_size: int = 1024
    ):  
        logging.info(f"Iterating reading table from: {table_dir}")
        filesystem = resolve_filesystem(table_dir)
        file_paths = filesystem.ls(table_dir)
        format = get_file_format(file_paths[0])
        reader = ReaderFactory.create_reader(format, file_paths, columns, filesystem.fs) 
        for batch_table in reader.iter_batches(batch_size):
            yield batch_table


    def write_table(
        self, 
        table: pa.Table, 
        table_dir: str,
        *,
        format: str = PARQUET,
        **kwargs
    ) -> str:
        os.path.abspath(table_dir)
        logging.info(f"Writting table to {table_dir}")
        filesystem = resolve_filesystem(table_dir)
        filesystem.mkdir(table_dir)
        WriterFactory.create_writer(format, filesystem.fs).write_table(
            table=table, 
            table_dir=table_dir,
            **kwargs
        )
        return table_dir

    def iter_writter(
        self,
        table_dir: str,
        *,
        format: str = PARQUET,
        **kwargs
    ):
        logging.info(f"Iterative writting table to: {table_dir}")
        filesystem = resolve_filesystem(table_dir)
        filesystem.mkdir(table_dir)
        writer = WriterFactory.create_iter_writer(format, filesystem.fs)
        writer.config(table_dir=table_dir, **kwargs)
        return writer
