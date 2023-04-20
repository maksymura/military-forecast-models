import os
import json

folder_path = "resources/alarms_v2"

output_folder = "resources/merged/"

output_file = output_folder + "alarms.json"

required_fields = [
    "address",
    "start",
    "end"
]


def read_and_transform():
    merged_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                gkey = os.path.splitext(filename)[0]

                filtered_values = [
                    {field: value[field] for field in required_fields}
                    for value in data
                ]

                merged_data[gkey] = filtered_values

    joined_data = {}
    for date in merged_data:
        joined_by_address = {}
        for alarm in merged_data[date]:
            if alarm["address"] in joined_by_address:
                joined_by_address[alarm["address"]] = {"alarms": joined_by_address[alarm["address"]]["alarms"]
                                                                 + [{"start": alarm["start"], "end": alarm["end"]}]}
            else:
                joined_by_address[alarm["address"]] = {"alarms": [{"start": alarm["start"], "end": alarm["end"]}]}
        joined_data[date] = joined_by_address

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")

    with open(output_file, 'w') as f:
        json.dump(joined_data, f, ensure_ascii=False, indent=4)
