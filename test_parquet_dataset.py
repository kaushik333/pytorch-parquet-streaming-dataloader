import time
from parquet_dataset import ParquetDataset
from torch.utils.data import DataLoader
import numpy as np
import unittest
import os

class ParquetDataloaderTest(unittest.TestCase):
  def setUp(self):
    self.file_path = "./test_dataset.parquet"
    self.batch_size = 20_000_000
    self.dataset = ParquetDataset(self.file_path, self.batch_size)
    self.dataloader = DataLoader(self.dataset, batch_size=None)

  def test_shuffle(self):
    assert not np.equal(next(iter(self.dataloader)).numpy(), next(iter(self.dataloader)).numpy()).all()
    print("Test passed | Shuffling test passed !")
    print("----------------------------------------")

  def test_iterate_full_dataloader(self):

    file_size = os.path.getsize(self.file_path)
    print(f"Size of the Parquet file: {(file_size/(1024**3))} GB")

    total_rows = len(self.dataset)
    num_rows = 0
    t0 = time.time()
    for idx, batch in enumerate(self.dataloader):
      num_rows += batch.shape[0]
    print(f"Took {time.time()-t0} seconds to iterate through entire dataset")
    assert total_rows == num_rows
    print("Test passed | Able to iterate through dataset fully !")
    print("--------------------------------------------------------")

  def test_batchsize_below_groupsize(self):
    assert self.dataset.chunk_size >= self.batch_size
    print("Test passed | Batch size is below parquet group size !")
    print("---------------------------------------------------------")

  def test_dataloader_batchsize(self):
    possible_bs1 = self.batch_size
    possible_bs2 = self.dataset.chunk_size % self.batch_size
    possible_bs3 = self.dataset.num_rows % self.batch_size
    possible_bs4 = (self.dataset.num_rows % self.dataset.chunk_size) % self.batch_size
    for _, batch in enumerate(self.dataloader):
      assert batch.size(0) == possible_bs1 or batch.size(0) == possible_bs2 or batch.size(0) == possible_bs3 or batch.size(0) == possible_bs4
    print("Test passed | Batchsize values are as expected !")
    print("---------------------------------------------------")
  
  def test_multiple_batchsizes(self):
    batchsizes = [999, 17327, 34989, 530654, 1019389]
    total_rows = len(self.dataset)
    for bs in batchsizes:
      self.batch_size = bs
      dataset = ParquetDataset(self.file_path, self.batch_size)
      dataloader = DataLoader(dataset, batch_size=None)
      num_rows = 0
      for _, batch in enumerate(self.dataloader):
        num_rows += batch.shape[0]
      assert num_rows == total_rows
      print(f"Test passed | Batch size = {bs} handles all edge cases !")
    print("------------------------------------------------------------")

if __name__ == "__main__":
  unittest.main()