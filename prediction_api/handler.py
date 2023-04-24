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
prediction_key = "prediction"

s3 = boto3.client('s3')


def predict():
    current_datetime = datetime.now()
    print(f"Now is {current_datetime}")

    hour = current_datetime.hour
    current_datetime_string = current_datetime.strftime('%Y-%m-%d')

    weather = get_weather(current_datetime_string)
    print("Fetched weather")

    tfidf = get_tfidf(current_datetime_string)
    print("Fetched tfidf")

    model = get_model()
    print("Loaded model")

    prediction = {}
    for city in cities_list():
        city_prediction = {}
        h = hour
        dt = current_datetime_string
        for i in range(0, 12):
            h = (h + 1) % 24
            dt = (dt, next_day(dt))[h + 1 > 24]
            prediction_df = get_df_for_prediction(city, weather, tfidf, dt, h)
            print("Created df for prediction")

            city_prediction = {**city_prediction,
                               **{number_to_hour_format(h): bool((model.predict(prediction_df)[0]) > 0.5)}}
        prediction = {**prediction, **{city: city_prediction}}

    prediction = {"last_prediction_time": str(current_datetime),
                  "regions_forecast": prediction}
    print("Created prediction")
    upload_predictions(prediction)
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

    return remove_duplicates(json.loads(weather_data))


def get_tfidf(current_datetime_string):
    tfidf_key = f"{tfidf_folder}/{format_date(current_datetime_string)}.json"
    try:
        tfidf_response = s3.get_object(Bucket=bucket_name, Key=tfidf_key)
    except:
        return 0
    tfidf_data = tfidf_response['Body'].read().decode('utf-8')
    vector = tf_idf_vector(tfidf_data)

    return np.mean(vector)


def get_df_for_prediction(city, weather, tfidf, current_date_time, hour):
    encoded_city_to_compare = get_region_number(city)

    features = ['city_encoded', 'day_of_week', 'day_of_year', "hour", "city_latitude", "city_longitude",
                "day_feelslike", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
                "day_precipprob", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']

    filtered_weather = [item for item in weather if get_region_number(item['city_address']) == encoded_city_to_compare][
        0]
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


def upload_predictions(prediction):
    json_string = json.dumps(prediction)
    json_bytes = json_string.encode('utf-8')
    file = io.BytesIO(json_bytes)
    file_name = f"{predictions_folder}/{prediction_key}.json"

    s3.upload_fileobj(file, bucket_name, file_name)
