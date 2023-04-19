try:
  import unzip_requirements
except ImportError:
  pass


import boto3
import pandas as pd
import pickle
import datetime
import json
import io

bucket_name = 'military-forecast'
weather_folder = 'weather_v3'
predictions_folder = 'predictions'
model_key = "model/knn.pickle"

s3 = boto3.client('s3')


def predict(event, ctx):
    current_datetime = datetime.datetime.now()
    print(f"Now is {current_datetime}")
    current_datetime_string = current_datetime.strftime('%Y-%m-%d')

    weather = get_weather(current_datetime_string)
    print("Fetched weather")

    prediction_df = get_df_for_prediction(weather)
    print("Created df for prediction")

    model = get_model()
    print("Loaded model")

    prediction = model.predict(prediction_df)
    print("Created prediction")

    upload_predictions(current_datetime_string, prediction.tolist())
    print("Uploaded prediction")

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({"prediction": prediction.tolist()})
    }

    return response


def get_weather(current_datetime_string):
    # TODO: replace 2022-02-24 with current_datetime_string
    weather_key = f"{weather_folder}/2022-02-24.json"
    weather_response = s3.get_object(Bucket=bucket_name, Key=weather_key)
    weather_data = weather_response['Body'].read().decode('utf-8')

    return weather_data


def get_df_for_prediction(weather):
    # TODO: use columns for bayes, use merged: current day weather and other data as model input
    columns = ['date', 'city', 'day_feelslikemax', 'day_feelslikemin',
               'day_feelslike', 'day_precipprob', 'day_snow',
               'day_snowdepth', 'day_windgust', 'day_windspeed', 'day_winddir',
               'day_pressure', 'day_cloudcover', 'day_visibility', 'day_severerisk',
               'day_preciptype', 'is_rus_holiday', 'is_ukr_holiday']
    values = [
        [1651363200, 10762219, 16.5, 7.0, 12.3, 0, 0.0, 0.0, 20.5, 7.9, 30.5, 1022.0, 55.2, 24.1, 10, 4496177, True,
         True]]

    df = pd.DataFrame(values, columns=columns)

    return df


def get_model():
    # TODO: fetch bayes model
    model_response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_file = model_response['Body'].read()
    model = pickle.loads(model_file)

    return model


def upload_predictions(current_datetime_string, prediction):
    json_string = json.dumps(prediction)
    json_bytes = json_string.encode('utf-8')
    file = io.BytesIO(json_bytes)
    file_name = f"{predictions_folder}/{current_datetime_string}.json"

    s3.upload_fileobj(file, bucket_name, file_name)
