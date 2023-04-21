from datetime import datetime, timedelta
import json


def tf_idf_vector(data):
    data = json.loads(data)
    unique_words = set(data["tfIdf"].keys())
    unique_words = sorted(list(unique_words))
    word_indices = {word: idx for idx, word in enumerate(unique_words)}

    vector = [0.0] * len(unique_words)
    for word, values in data["tfIdf"].items():
        idx = word_indices[word]
        vector[idx] = values["tfidf"]

    return vector


def is_ukr_holiday(day_of_year):
    with open("resources/holidays.json", 'r') as f:
        holidays_data = json.load(f)
    date_obj = datetime(year=datetime.now().year, month=1, day=1) + timedelta(days=day_of_year - 1)
    return date_obj.strftime('%d-%m') in holidays_data['ukr'].keys()


def is_rus_holiday(day_of_year):
    with open("resources/holidays.json", 'r') as f:
        holidays_data = json.load(f)
    date_obj = datetime(year=datetime.now().year, month=1, day=1) + timedelta(days=day_of_year - 1)
    return date_obj.strftime('%d-%m') in holidays_data['rus'].keys()


def get_day_of_week(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.weekday()


def get_day_of_year(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.timetuple().tm_yday


def get_month(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.month


def get_region_number(city):
    if city == "Simferopol":
        return 1
    elif city == "Vinnytsia":
        return 2
    elif city == "Lutsk":
        return 3
    elif city == "Dnipro":
        return 4
    elif city == "Donetsk":
        return 5
    elif city == "Zhytomyr":
        return 6
    elif city == "Uzhgorod":
        return 7
    elif city == "Zaporozhye":
        return 8
    elif city == "Ivano-Frankivsk":
        return 9
    elif city == "Kyiv":
        return 10
    elif city == "Kropyvnytskyi":
        return 11
    elif city == "Luhansk":
        return 12
    elif city == "Lviv":
        return 13
    elif city == "Mykolaiv":
        return 14
    elif city == "Odesa":
        return 15
    elif city == "Poltava":
        return 16
    elif city == "Rivne":
        return 17
    elif city == "Sumy":
        return 18
    elif city == "Ternopil":
        return 19
    elif city == "Kharkiv":
        return 20
    elif city == "Kherson":
        return 21
    elif city == "Khmelnytskyi":
        return 22
    elif city == "Cherkasy":
        return 23
    elif city == "Chernivtsi":
        return 24
    elif city == "Chernihiv":
        return 25
    print("HERE " + city)
    return -1


def format_date(date_string: str) -> str:
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    formatted_date_string = f"{date_object.year}-{date_object.month}-{date_object.day}"

    return formatted_date_string


def number_to_hour_format(hour):
    return f"{hour:02d}:00"
