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
