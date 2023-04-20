import os
import json
from datetime import datetime

folder_path = "resources/isw/tfidf"
output_folder = "resources/merged/"
output_file = output_folder + "tfidf.json"


def read_and_transform():
    merged_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                gkey = datetime.strptime(os.path.splitext(filename)[0], "%Y-%m-%d").strftime("%Y-%m-%d")
                del data["date"]
                merged_data[gkey] = data

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
