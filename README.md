# Larger than memory Pytorch dataloader for Parquet files.

### Instructions to run

Step 1: First install dependencies:
```python
    conda create --name torch25 --yes
    conda activate torch25
    pip install -r requirements.txt
```

Step 2: Create a toy dataset to test the `ParquetDataset` class (available in `parquet_dataset.py`) -- Creates a dataset of size ~7.5GB. In the file `create_test_dataset.py`, increase the `num_rows` variable (or `num_columns` variable) to increase size of dataset even further.
```python
    python create_test_dataset.py
```

Step 3: Test the `ParquetDataset` class (available in `parquet_dataset.py`) using a series of different tests using python's `unittest` module`.
```python
    python -m unittest discover
```

### Notes
- `ParquetDataset` has been written to handle multiple workers to load the data from the parquet file concurrently without overlap of data. This is especially useful for larger than memory scenarios where the entire parquet file cannot be loaded into the memory.
- `num_workers` has been set to 0 since during development on Windows machine, there seems to be a bug which causes any `num_workers` > 0 to cause a file-reading error. `num_workers` > 0 has been tried and successfully tested on google-colab. Please refer [this Google colab notebook](https://colab.research.google.com/drive/1ZoGWcJkbNFlT_XDH4_g7hgKQTnsiZ_yE?usp=sharing) where 8 workers have been used. Hence this should work on your linux/mac system.
- Can further improve performance using the dask library which will scale up the parquet reading across multiple systems in a distributed setting, which is the drawback of the current approach.
- There are some existing implementations like [this](https://github.com/KamWithK/PyParquetLoaders/blob/master/PyTorchLoader.py) which use `to_batches()` functionality of PyArrow datasets, but these require storing all batches in memory which doesnt suite datasets that dont fit in memory which are typically 100s of GBs.
- Drawbacks: 
    - This code works when parquet has small sized chunks. If you have very large chunks and you do not have enough memory (RAM), using multiple workers will not make any difference and you will see the code crash. The inherent problem is that each worker loads one chunk (row group in parquet file) at a time. Thus, holding N chunks (corresponding to N workers) concurrently needs larger memory size.
