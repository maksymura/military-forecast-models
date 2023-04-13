import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit


def train():
    # 1. Load the data
    data = pd.read_csv("resources/dataset/final_data.csv", parse_dates=['date', 'alarm_start', 'alarm_end', 'time'])
    # 2. Preprocess the data
    # Create a binary label for whether an alarm occurred
    data['vector'] = data['vector'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split(',')], dtype=np.float32) if isinstance(x, str) else x)
    print("vector1")
    # Apply the np.mean function to the 'vector' column
    # Compute the mean of the 'vector' column
    data['vector_mean'] = data['vector'].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan)

    # Impute missing values in 'vector_mean'
    imputer = SimpleImputer()
    data['vector_mean'] = imputer.fit_transform(data[['vector_mean']])
    data['alarm_occurred'] = np.where((data['alarm_start'].notna()) & (data['alarm_end'].notna()), 1, 0)
    # Encode the city names
    le = LabelEncoder()
    data['city_encoded'] = le.fit_transform(data['city'])

    # Extract features from the date
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    data['month'] = data['date'].dt.month

    # Extract features and target
    features = ['city_encoded', 'day_of_week', 'day_of_year', 'month',
    "day_feelslikemin",
    "day_sunriseEpoch",
    "day_sunsetEpoch",
    "city_latitude",
    "city_longitude",
    "city_tzoffset",
    "day_feelslike",
    "day_precipprob",
    "day_snow",
    "day_snowdepth",
    "day_windgust",
    "day_winddir",
    "day_pressure",
    "day_cloudcover",
    "day_severerisk",
    'is_rus_holiday',
    'is_ukr_holiday',
    'vector_mean'
                ]
    X = data[features]
    y = data['alarm_occurred']

    # Split data into training and testing sets
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 3. Train the Logistic Regression model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Define hyperparameters to search over
    param_grid = {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga']
    }

    # Perform grid search using cross-validation
    clf = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)

    print("Best hyperparameters:", clf.best_params_)
    print("Best accuracy:", clf.best_score_)

    # Check accuracy on the test set using the best model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion matrix:")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
