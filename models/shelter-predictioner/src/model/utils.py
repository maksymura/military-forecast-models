import json
from collections import OrderedDict
import json
import re
import nltk
from nltk.corpus import stopwords
file_path = "resources/merged/tfidf.json"


def tfIdfVector():
    with open(file_path, 'r') as file:
        data = json.load(file, object_pairs_hook=OrderedDict)

    stop_words = set(stopwords.words('english'))
    unique_words = set()
    for date, tf_idf_data in data.items():
        unique_words |= set(tf_idf_data["tfIdf"].keys())

    unique_words = sorted(list(unique_words))
    word_indices = {word: idx for idx, word in enumerate(unique_words)}

    vectors = {}

    for date, tf_idf_data in data.items():
        vector = [0.0] * len(unique_words)
        for word, values in tf_idf_data["tfIdf"].items():
            idx = word_indices[word]
            vector[idx] = values["tfidf"]
        vectors[date] = vector

    with open('resources/dataset/vector.json', 'w') as outfile:
        json.dump(vectors, outfile, indent=2)


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
