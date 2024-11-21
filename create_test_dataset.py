import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Define the number of rows and columns
num_rows = 100_000_000  # Adjust this number to achieve the desired file size
num_columns = 10
chunk_size = 20_485_760  # set it to a number which can fit in your memory

filename = "test_dataset.parquet"

# Function to generate a chunk of data
def generate_chunk(start, size, num_columns):
    data = {f'col_{i}': np.random.rand(size) for i in range(num_columns)}
    return pd.DataFrame(data)

# Create a Parquet writer
schema = pa.schema([pa.field(f'col_{i}', pa.float64()) for i in range(num_columns)])
# writer = pq.ParquetWriter(filename, schema)
# Write data in chunks
with pq.ParquetWriter(filename, schema) as writer:
    for start in range(0, num_rows, chunk_size):
        print(start)
        chunk = generate_chunk(start, min(chunk_size, num_rows - start), num_columns)
        table = pa.Table.from_pandas(chunk)
        writer.write_table(table, row_group_size=chunk_size)
# Close the writer
# writer.close()

print("Parquet file created successfully!")

file_size = os.path.getsize(filename)
print(f"Size of the Parquet file: {(file_size/(1024**3))} GB")