import os
import json

folder_path = "resources/weather_v3"
output_folder = "resources/merged/"
output_file = output_folder + "weather.json"

required_fields = [
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
    "hour_feelslike"
]

def hashable_dict(d):
    return tuple(sorted((key, tuple(value) if isinstance(value, list) else value) for key, value in d.items()))

def remove_duplicates(json_list):
    seen = set()
    unique_json_list = []

    for item in json_list:
        item_tuple = hashable_dict(item)
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_json_list.append(item)

    return unique_json_list


def read_and_transform():
    merged_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = remove_duplicates(json.load(f))
                for obj in data:
                    for field in obj:
                        filtered_value = {field: obj[field] for field in required_fields}
                    city = obj["city_address"]
                    hour = obj["hour_datetime"]
                    city_data = {
                        city: {hour: filtered_value}
                    }
                    if filename[:-5] in merged_data:
                        if city in merged_data[filename[:-5]]:
                            city_data = {
                                city: {**merged_data[filename[:-5]][city], **{hour: filtered_value}}
                            }
                        merged_data[filename[:-5]] = {**merged_data[filename[:-5]], **city_data}
                    else:
                        merged_data[filename[:-5]] = city_data

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
