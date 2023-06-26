import sys
import sklearn
import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')
    
    return dicts

def apply_model(input_file, output_file, year, month):
    print(f'Reading data for {year}-{month}')
    df = read_data(input_file)

    print(f'Preparing dictionaries ... ')
    dicts = prepare_dictionaries(df)
    X_val = dv.transform(dicts)

    print(f'Running predict ... ')
    y_pred = model.predict(X_val)

    print(f'Saving results in {output_file}')
    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predictions'] = y_pred

    mean_predicted_duration = round(df_result.loc[:, 'predictions'].mean(), 2)
    print(f'mean predicted duration: {mean_predicted_duration}')

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def run():
    year = int(sys.argv[1]) #2022
    month = int(sys.argv[2]) #2
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'gs://taxi-data-mlflow-artifacts/deployment-scoring-artifacts/result/yellow/{year:04d}-{month:02d}.parquet'
    # f'./output/yellow/{year:04d}-{month:02d}.parquet'

    apply_model(input_file=input_file, output_file=output_file, year=year, month=month)

if __name__ == '__main__':
    run()