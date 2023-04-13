import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def train():
    # Load data
    data = pd.read_csv('resources/dataset/final_data.csv')

    data['vector']= data['vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    vector_df = pd.DataFrame(data['vector'].tolist())

    X = pd.concat([data[['day_tempmax', 'day_tempmin', 'day_humidity', 'day_precip', 'day_windgust', 'day_windspeed',
                         'day_winddir', 'day_cloudcover', 'day_visibility']], vector_df], axis=1)

    X.columns = X.columns.astype(str)

    data['has_alarm'] = data['alarm_start'].notna().astype(int)
    y = data['has_alarm']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, data, scaler


def predict_alarm(city, date, model, data, scaler):
    city_data = data[data['city'] == city]
    city_data['date'] = pd.to_datetime(city_data['date'])
    city_data = city_data[city_data['date'].dt.date == pd.to_datetime(date).date()]

    if city_data.empty:
        print("No data available for the given city and date.")
        return

    X_city = pd.concat([city_data[['day_tempmax', 'day_tempmin', 'day_humidity', 'day_precip', 'day_windgust',
                                   'day_windspeed', 'day_winddir', 'day_cloudcover', 'day_visibility']],
                        city_data['vector'].apply(pd.Series)], axis=1)
    X_city.columns = X_city.columns.astype(str)
    X_city_scaled = scaler.transform(X_city)

    alarm_probabilities = model.predict_proba(X_city_scaled)[:, 1]
    hours = city_data['time']

    for hour, prob in zip(hours, alarm_probabilities):
        print(f"{hour}: {prob:.2f}")


