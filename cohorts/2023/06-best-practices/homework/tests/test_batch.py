from datetime import datetime
import batch
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():
    input_data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    input_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    input_df = pd.DataFrame(input_data, columns=input_columns)

    categorical = ['PULocationID', 'DOLocationID']

    actual_result = batch.prepare_data(input_df, categorical, "tpep_pickup_datetime", "tpep_dropoff_datetime").to_dict()

    output_data = [
        ('-1','-1',pd.to_datetime('2022-01-01 01:02:00'),pd.to_datetime('2022-01-01 01:10:00'),8.0),
	    ('1','-1',pd.to_datetime('2022-01-01 01:02:00'),pd.to_datetime('2022-01-01 01:10:00'),8.0),
	    ('1','2', pd.to_datetime('2022-01-01 02:02:00'),pd.to_datetime('2022-01-01 02:03:00'),1.0)
    ]
    output_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_result = pd.DataFrame(output_data, columns=output_columns).to_dict()

    assert expected_result == actual_result
