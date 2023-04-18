try:
  import unzip_requirements
except ImportError:
  pass


import boto3
import pandas as pd
import pickle
import datetime

s3 = boto3.client('s3')

bucket_name = 'military-forecast'
weather_key = 'weather_v3/2022-02-24.json'
model_key = "model/knn.pickle"


# TODO:
#  1. Update get_prediction_df to work with real weather data from get_weather and bayes model
#  2. Save prediction to S3

def predict(event, ctx):
    current_datetime = datetime.datetime.now()
    print(f"Now is {current_datetime}")

    prediction_df = get_df_for_prediction()
    print("Created df for prediction")

    model = get_model()
    print("Loaded model")

    prediction = model.predict(prediction_df)

    return {
        'prediction': prediction[0]
    }


def get_weather():
    weather_response = s3.get_object(Bucket=bucket_name, Key=weather_key)
    weather_data = weather_response['Body'].read().decode('utf-8')

    return weather_data


def get_df_for_prediction():
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
    model_response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_file = model_response['Body'].read()
    model = pickle.loads(model_file)

    return model

