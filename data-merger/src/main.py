from data_tranformers import weather_transformer as wt, alarm_transformer as at, dataset_generator as awm, iws_transformer as it
from model.naive_bayes import train

from s3_service import get_s3_data

get_s3_data("alarms_v2/")
get_s3_data("weather_v3/")
get_s3_data("isw/tfidf/")
wt.read_and_transform()
at.read_and_transform()
it.read_and_transform()
awm.dataset()

# train()

