#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd

def prepare_data(df, categorical_features, pickup_datetime_colname, dropoff_datetime_colname):
    df['duration'] = df[dropoff_datetime_colname] - df[pickup_datetime_colname]
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical_features] = df[categorical_features].fillna(-1).astype('int').astype('str')

    return df

def read_data(filename):
    options = {
        'client_kwargs': {
            'endpoint_url': "http://localhost:4566" #S3_ENDPOINT_URL
        }
    }
    df = pd.read_parquet(filename, storage_options=options)

    return df


def save_data(filename, df_output):
    options = {
        'client_kwargs': {
            'endpoint_url': "http://localhost:4566" #S3_ENDPOINT_URL
        }
    }

    df_output.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year: int, month: int):
    # input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(f'Loading Model ... ')
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    print(f'Reading data from {input_file}')
    df = read_data(input_file)

    print(f'Preparing data ...')
    df = prepare_data(df, categorical, "tpep_pickup_datetime", "tpep_dropoff_datetime")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted sum duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    print(f'Saving Results to {output_file} ')
    save_data(output_file, df_result)

    # options = {
    #     'client_kwargs': {
    #         'endpoint_url': "http://localhost:4566" #S3_ENDPOINT_URL
    #     }
    # }
    # df = pd.read_parquet("s3://nyc-duration/in/2022-01.parquet", storage_options=options)

    # print(df['predicted_duration'].sum())

if __name__ == '__main__':
    main(year = int(sys.argv[1]), month = int(sys.argv[2]))