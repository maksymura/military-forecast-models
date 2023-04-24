import json
import boto3
import pickle
import datetime
import pandas as pd
from train_model import train

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')
bucket_name = 'military-forecast'
file = 'model/naive_bayes_v1.pkl'
date_file = 'model/train_time.json'


def rebuild_model():
    # obj = s3.get_object(Bucket='bucket', Key='key')
    # data = pd.read_csv(io.BytesIO(obj['Body'].read()))
    data = pd.read_csv("path/to/final_data_new.csv", parse_dates=['date', 'alarm_start', 'alarm_end', 'time'])
    model = train(data)
    model_bytes = pickle.dumps(model)
    s3_resource.Object(bucket_name, file).put(Body=model_bytes)

    json_date = {'date': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}
    data_file = s3_resource.Object(bucket_name, date_file)
    data_file.put(
        Body=(bytes(json.dumps(json_date, indent=4).encode('UTF-8')))
    )
