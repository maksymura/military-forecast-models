import numpy as np

try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import pandas as pd
import pickle
import io
from utils import *

bucket_name = 'military-forecast'
weather_folder = 'weather_v3'
tfidf_folder = 'isw/tfidf'
predictions_folder = 'predictions'
model_key = "model/naive_bayes_v1.pkl"

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIATV5NCJMWJHKI44PA',
    aws_secret_access_key='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'
)


def predict(city, date):
    # current_datetime = datetime.datetime.now()
    current_datetime = date
    print(f"Now is {current_datetime}")
    current_datetime_string = current_datetime.strftime('%Y-%m-%d')

    weather = get_weather(current_datetime_string)
    print("Fetched weather")

    tfidf = get_tfidf(current_datetime_string)
    print("Fetched tfidf")


    model = get_model()
    print("Loaded model")

    prediction = {}
    for i in range(0, 24):
        prediction_df = get_df_for_prediction(city, weather, tfidf, current_datetime_string, i)
        print("Created df for prediction")

        prediction = {**prediction, **{number_to_hour_format(i): int(model.predict(prediction_df)[0])}}

    print("Created prediction")
    upload_predictions(current_datetime_string, prediction)
    print("Uploaded prediction")

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({"prediction": prediction})
    }

    return response


def get_weather(current_datetime_string):
    weather_key = f"{weather_folder}/{current_datetime_string}.json"
    weather_response = s3.get_object(Bucket=bucket_name, Key=weather_key)
    weather_data = weather_response['Body'].read().decode('utf-8')

    return weather_data


def get_tfidf(current_datetime_string):
    tfidf_key = f"{tfidf_folder}/{format_date(current_datetime_string)}.json"
    tfidf_response = s3.get_object(Bucket=bucket_name, Key=tfidf_key)
    tfidf_data = tfidf_response['Body'].read().decode('utf-8')
    vector = tf_idf_vector(tfidf_data)

    return np.mean(vector)


def get_df_for_prediction(city, weather, tfidf, current_date_time, hour):
    weather = json.loads(weather)
    encoded_city_to_compare = get_region_number(city)

    features = ['city_encoded', 'day_of_week', 'day_of_year', "hour", "city_latitude", "city_longitude",
                "day_feelslike", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
                "day_precipprob", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']

    filtered_weather = [item for item in weather
                        if get_region_number(item['city_address']) == encoded_city_to_compare][0]
    merged_data = {**filtered_weather, **{"city_encoded": encoded_city_to_compare,
                                          "is_rus_holiday": is_rus_holiday(get_day_of_year(current_date_time)),
                                          "is_ukr_holiday": is_ukr_holiday(get_day_of_year(current_date_time)),
                                          "vector_mean": tfidf,
                                          "day_of_week": get_day_of_week(current_date_time),
                                          "day_of_year": get_day_of_year(current_date_time),
                                          "hour": hour}}

    values = [[merged_data[item] for item in features]]
    df = pd.DataFrame(values, columns=features)
    return df


def get_model():
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
