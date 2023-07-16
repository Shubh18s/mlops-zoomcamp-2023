import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = "s3://nyc-duration/in/2022-01.parquet"
options = {
    'client_kwargs': {
        'endpoint_url': "http://localhost:4566"
    }
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

dir_path = os.path.dirname(os.path.realpath("../batch.py"))
file_path = os.path.join(dir_path,'batch.py')
print(dir_path)

year = 2022
month = 1
os.system(f'python {file_path} {year} {month}') # Add 3rd param model_path to invoke batch.py from  here


# options = {
#         'client_kwargs': {
#             'endpoint_url': "http://localhost:4566" #S3_ENDPOINT_URL
#         }
#     }
# df = pd.read_parquet("s3://nyc-duration/in/2022-01.parquet", storage_options=options)

# print(df['predictions'].sum())