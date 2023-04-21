try:
  import unzip_requirements
except ImportError:
  pass

from sklearn.preprocessing import LabelEncoder
import boto3
import pandas as pd
import pickle
import datetime
import json
import io

bucket_name = 'military-forecast'
weather_folder = 'weather_v3'
tfidf_folder = 'isw/tfidf'
predictions_folder = 'predictions'
model_key = "model/bias.pickle"

s3 = boto3.client('s3')


def predict(event, ctx):
    current_datetime = datetime.datetime.now()
    print(f"Now is {current_datetime}")
    current_datetime_string = current_datetime.strftime('%Y-%m-%d')

    weather = get_weather(current_datetime_string)
    print("Fetched weather")

    tfidf = get_tfidf(current_datetime_string)
    print("Fetched tfidf")

    prediction_df = get_df_for_prediction(weather, tfidf, current_datetime_string)
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
    weather_key = f"{weather_folder}/{current_datetime_string}.json"
    weather_response = s3.get_object(Bucket=bucket_name, Key=weather_key)
    weather_data = weather_response['Body'].read().decode('utf-8')

    return weather_data

def get_tfidf(current_datetime_string):
    tfidf_key = f"{tfidf_folder_folder}/{format_date(current_datetime_string) }.json"
    tfidf_response = s3.get_object(Bucket=bucket_name, Key=tfidf_key)
    tfidf_data = tfidf_response['Body'].read().decode('utf-8')
    tf_idf_vector(tfidf_data).apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan)

    return tfidf_data


def get_df_for_prediction(city, weather, tfidf, current_date_time):
    le = LabelEncoder()

    encoded_city_to_compare = le.fit_transform(city)

    filtered_weather = [item for item in payload
                        if encode_city_address(item['city_address']) == encoded_city_to_compare
                        and not (encode_city_address(item['city_address']) in seen or seen.add(
            encode_city_address(item['city_address'])))]

    df = pd.Series({
        'city_encoded': encoded_city_to_compare,
        'day_of_week': weather[day_severerisk],
        'day_of_year': weather[day_severerisk],
        'month': weather[day_severerisk],
        'day_feelslikemin': filtered_weather[day_severerisk],
        'day_sunriseEpoch': filtered_weather[day_severerisk],
        'day_sunsetEpoch': filtered_weather[day_severerisk],
        'city_latitude': filtered_weather[day_severerisk],
        'city_longitude': filtered_weather[day_severerisk],
        'city_tzoffset': filtered_weather[day_severerisk],
        'day_feelslike': filtered_weather[day_severerisk],
        'day_precipprob': filtered_weather[day_severerisk],
        'day_snow': filtered_weather[day_severerisk],
        'day_snowdepth': filtered_weather[day_severerisk],
        'day_windgust': filtered_weather[day_severerisk],
        'day_winddir': filtered_weather[day_severerisk],
        'day_pressure': filtered_weather[day_severerisk],
        'day_cloudcover': filtered_weather[day_severerisk],
        'day_severerisk': filtered_weather[day_severerisk],
        'is_rus_holiday': is_rus_holiday(current_date_time),
        'is_ukr_holiday': is_ukr_holiday(current_date_time),
        'vector_mean': tfidf
    }).to_frame().T

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


def format_date(date_string: str) -> str:
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    formatted_date_string = f"{date_object.year}-{date_object.month}-{date_object.day}"

    return formatted_date_string

def tf_idf_vector(data):
    unique_words = set(data["tfIdf"].keys())
    unique_words = sorted(list(unique_words))
    word_indices = {word: idx for idx, word in enumerate(unique_words)}

    vector = [0.0] * len(unique_words)
    for word, values in data["tfIdf"].items():
        idx = word_indices[word]
        vector[idx] = values["tfidf"]

    return vector

def is_ukr_holiday(date):
    with open("resources/holidays.json", 'r') as f:
        holidays_data = json.load(f)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    return date_obj.strftime('%d-%m') in holidays_data['ukr'].keys()


def is_rus_holiday(date):
    with open("resources/holidays.json", 'r') as f:
        holidays_data = json.load(f)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    return date_obj.strftime('%d-%m') in holidays_data['rus'].keys()

def get_year(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.year

def get_month(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.month

def get_day(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.day