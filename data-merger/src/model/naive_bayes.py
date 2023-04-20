import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def train():
    # 1. Load the data
    data = pd.read_csv("resources/dataset/final_data.csv", parse_dates=['date', 'alarm_start', 'alarm_end', 'time'])

    # 2. Preprocess the data
    # Create a binary label for whether an alarm occurred
    data['vector'] = data['vector'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split(',')], dtype=np.float32) if isinstance(x, str) else x)
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
    features = ['city_encoded', 'day_of_week', 'day_of_year', 'month', "day_feelslikemin", "day_sunriseEpoch",
                "day_sunsetEpoch", "city_latitude", "city_longitude", "city_tzoffset", "day_feelslike",
                "day_precipprob", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
                "day_cloudcover", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']
    X = data[features]
    y = data['alarm_occurred']

    # Split data into training and testing sets using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Naive Bayes model
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])

        # Define hyperparameters to search over
        param_grid = {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }

        # Perform grid search using cross-validation
        clf = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)

        # Print the best hyperparameters and their corresponding accuracy
        print("Best hyperparameters:", clf.best_params_)
        print("Best accuracy:", clf.best_score_)

        # Check accuracy on the test set using the best model
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))

        feature_importance(clf, X_train, y_train, X_test, y_test, features)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("Confusion matrix:")
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Positives:", tp)


def feature_importance(clf, X_train, y_train, X_test, y_test, features):
        base_accuracy = accuracy_score(y_test, clf.predict(X_test))

        importances = []

        for feature in features:
            X_train_dropped = X_train.drop(columns=[feature])
            X_test_dropped = X_test.drop(columns=[feature])

            clf.fit(X_train_dropped, y_train)
            dropped_accuracy = accuracy_score(y_test, clf.predict(X_test_dropped))

            importance = base_accuracy - dropped_accuracy
            importances.append((feature, importance))

        importances.sort(key=lambda x: x[1], reverse=True)
        return importances

def plot_feature_importance(importances, top_n=20):
    # Display the top features
    top_features = importances[:top_n]
    feature_names, importance_values = zip(*top_features)

    # Create the bar chart
    plt.figure(figsize=(12, 10))
    plt.bar(range(top_n), importance_values, align='center')
    plt.xticks(range(top_n), feature_names, rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.show()