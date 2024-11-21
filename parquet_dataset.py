import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import os
import multiprocessing as mp

class ParquetDataset(IterableDataset):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_row_groups = self.parquet_file.metadata.num_row_groups
        self.chunk_size = self.parquet_file.read_row_group(0).num_rows
        self.num_rows = self.parquet_file.metadata.num_rows

    def __len__(self):
      return pq.ParquetFile(self.file_path).metadata.num_rows

    def __iter__(self):

      worker_info = get_worker_info()
      if worker_info is None:
        worker_id = 0
        num_workers = 1
      else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        if num_workers > self.num_row_groups:
          num_workers = self.num_row_groups
          if worker_id >= num_workers:
            return

      num_rowgroups_per_worker = np.floor(self.num_row_groups/num_workers)
      row_group_ranges = []
      for idx in range(num_workers):
        if (idx+1)*num_rowgroups_per_worker <= self.num_row_groups:
          row_group_ranges.append([idx*num_rowgroups_per_worker, (idx+1)*num_rowgroups_per_worker])
        else:
          row_group_ranges.append([idx*num_rowgroups_per_worker, self.num_row_groups+1])
      if row_group_ranges[-1][1] != self.num_row_groups:
        row_group_ranges[-1][1] = self.num_row_groups

      for row_group_index in range(int(row_group_ranges[worker_id][0]), int(row_group_ranges[worker_id][1])):
          row_group = pq.ParquetFile(self.file_path).read_row_group(row_group_index)
          df = row_group.to_pandas()
          df = df.sample(frac=1).reset_index(drop=True)
          num_rows = len(df)
          for start_idx in range(0, num_rows, self.batch_size):
              batch = df.iloc[start_idx:start_idx + self.batch_size]
              yield torch.tensor(batch.values, dtype=torch.float32)

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    file_path = "./test_dataset.parquet"
    batch_size = 20_000_000
    dataset = ParquetDataset(file_path, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    file_size = os.path.getsize(file_path)
    print(f"Size of the Parquet file: {(file_size/(1024**3))} GB")

    for idx, batch in enumerate(dataloader):
      print(batch.shape)