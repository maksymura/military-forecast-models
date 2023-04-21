import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_region_number


data = pd.read_csv("resources/dataset/final_data.csv",
        parse_dates=['date', 'alarm_start', 'alarm_end', 'time'])
features = ['city_encoded', 'day_of_week', 'day_of_year', 'month', "day_feelslikemin", "day_sunriseEpoch",
            "day_sunsetEpoch", "city_latitude", "city_longitude", "city_tzoffset", "day_feelslike",
            "day_precipprob", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
            "day_cloudcover", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']

# features = ['city_encoded', 'day_of_week', 'day_of_year', "city_latitude", "city_longitude", "city_tzoffset",
#             "day_feelslike", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
#                 "day_precipprob", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']

# 2. Preprocess the data
# Create a binary label for whether an alarm occurred
data['vector'] = data['vector'].apply(
    lambda x: np.array([float(i) for i in x.strip('[]').split(',')], dtype=np.float32) if isinstance(x, str) else x)
data['vector_mean'] = data['vector'].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan)

# Impute missing values in 'vector_mean'
imputer = SimpleImputer()
data['vector_mean'] = imputer.fit_transform(data[['vector_mean']])

data['alarm_occurred'] = np.where((data['alarm_start'].notna()) & (data['alarm_end'].notna()), 1, 0)

# Encode the city names
data['city_encoded'] = data['city'].apply(lambda x: get_region_number(str(x)))

# Extract features from the date
data['day_of_week'] = data['date'].dt.dayofweek
data['day_of_year'] = data['date'].dt.dayofyear
data['month'] = data['date'].dt.month

# Compute the correlation matrix
corr_matrix = data[features].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()
