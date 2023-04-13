import json
import csv
from datetime import datetime

from dateutil.parser import parse

from model.utils import tfIdfVector

weather_file_path = "resources/merged/weather.json"
alarms_file_path = "resources/merged/alarms.json"
iws_file_path = "resources/merged/tfidf.json"
vector_file_path = "resources/dataset/vector.json"
csv_output_file = "resources/dataset/dataset.csv"
output_filename = "resources/dataset/final_data.csv"


def dataset():
    json_output_file = "resources/dataset/dataset.json"

    merged_data = {}
    with open(weather_file_path, 'r') as f:
        weather_data = json.load(f)

    with open(alarms_file_path, 'r') as f:
        alarms_data = json.load(f)

    dates = list(set(list(weather_data.keys()) + list(alarms_data.keys())))
    for date in dates:
        if date in weather_data:
            weather = weather_data[date]
        else:
            weather = {}
        if date in alarms_data:
            alarms = alarms_data[date]
        else:
            alarms = {}
        cities = list(set(list(alarms.keys()) + list(weather.keys())))
        date_data = {}
        for city in cities:
            if city in alarms:
                alarm_dict = alarms[city]
            else:
                alarm_dict = {"alarms": []}

            if city in weather:
                weather_dict = {"weather": weather[city]}
            else:
                weather_dict = {"weather": {}}

            date_data[city] = {**weather_dict, **alarm_dict}
        merged_data[date] = date_data

    with open(json_output_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    tfIdfVector()
    with open(json_output_file, 'r') as f:
        data = json.load(f)
    rows = []
    for date, city_data in data.items():
        for city, city_weather in city_data.items():
            for time, weather_data in city_weather['weather'].items():
                row = {'Date': date, 'City': city, 'Time': time}
                row.update(weather_data)
                rows.append(row)

    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['date',
                         'city',
                         'time',
                         "day_feelslikemax",
                         "day_feelslikemin",
                         "day_sunriseEpoch",
                         "day_sunsetEpoch",
                         "day_description",
                         "city_latitude",
                         "city_longitude",
                         "city_address",
                         "city_timezone",
                         "city_tzoffset",
                         "day_feelslike",
                         "day_precipprob",
                         "day_snow",
                         "day_snowdepth",
                         "day_windgust",
                         "day_windspeed",
                         "day_winddir",
                         "day_pressure",
                         "day_cloudcover",
                         "day_visibility",
                         "day_severerisk",
                         "day_conditions",
                         "day_icon",
                         "day_source",
                         "day_preciptype",
                         "day_stations",
                         "hour_icon",
                         "hour_source",
                         "hour_stations",
                         "hour_feelslike",
                         'is_rus_holiday',
                         'is_ukr_holiday',
                         'alarm_start',
                         'alarm_end'])

        for date, cities in data.items():
            for city, city_data in cities.items():
                for time, weather_data in city_data['weather'].items():
                    alarm_start = ''
                    alarm_end = ''
                    for alarm in city_data['alarms']:
                        alarm_start_date = alarm['start'].split()[0];
                        alarm_end_date = alarm['end'].split()[0];

                        time_h = time.split(':')[0]
                        alarm_start_h = alarm['start'].split()[1].split(':')[0]
                        alarm_end_h = alarm['end'].split()[1].split(':')[0]
                        if (
                                alarm_start_date == date and alarm_end_date == date and alarm_start_h <= time_h <= alarm_end_h) \
                                or (alarm_start_date != date and alarm_end_date == date and time_h <= alarm_end_h) \
                                or (alarm_start_date < date < alarm_end_date):
                            alarm_start = alarm['start']
                            alarm_end = alarm['end']
                    writer.writerow([
                        date,
                        city,
                        time,
                        weather_data['day_feelslikemax'],
                        weather_data['day_feelslikemin'],
                        weather_data['day_sunriseEpoch'],
                        weather_data['day_sunsetEpoch'],
                        weather_data['day_description'],
                        weather_data['city_latitude'],
                        weather_data['city_longitude'],
                        weather_data['city_address'],
                        weather_data['city_timezone'],
                        weather_data['city_tzoffset'],
                        weather_data['day_feelslike'],
                        weather_data['day_precipprob'],
                        weather_data['day_snow'],
                        weather_data['day_snowdepth'],
                        weather_data['day_windgust'],
                        weather_data['day_windspeed'],
                        weather_data['day_winddir'],
                        weather_data['day_pressure'],
                        weather_data['day_cloudcover'],
                        weather_data['day_visibility'],
                        weather_data['day_severerisk'],
                        weather_data['day_conditions'],
                        weather_data['day_icon'],
                        weather_data['day_source'],
                        weather_data['day_preciptype'],
                        weather_data['day_stations'],
                        weather_data['hour_icon'],
                        weather_data['hour_source'],
                        weather_data['hour_stations'],
                        weather_data['hour_feelslike'],
                        is_rus_holiday(date),
                        is_ukr_holiday(date),
                        alarm_start,
                        alarm_end
                    ])

    with open(vector_file_path, "r") as json_file:
        vectors = json.load(json_file)

    with open(csv_output_file, "r") as csv_file, open(output_filename, "w") as json_output_file:
        csv_reader = csv.reader(csv_file)
        csv_writer = csv.writer(json_output_file)

        header = next(csv_reader)
        header.append("vector")
        csv_writer.writerow(header)

        for row in csv_reader:
            date_time_str = row[0]

            if len(row[2]) > 0:
                date_time_str += f" {row[2]}"

            date = parse(date_time_str).strftime("%Y-%m-%d")

            if date in vectors:
                row.append(vectors[date])
            else:
                row.append(None)

            csv_writer.writerow(row)


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
