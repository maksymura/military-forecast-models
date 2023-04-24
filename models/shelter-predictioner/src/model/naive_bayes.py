import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from utils import get_region_number
from sklearn.model_selection import StratifiedShuffleSplit


def train():
    # 1. Load the data
    data = pd.read_csv("resources/dataset/final_data.csv", parse_dates=['date', 'alarm_start', 'alarm_end', 'time'])

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
    data['hour'] = data['time'].dt.hour
    data['month'] = data['date'].dt.month

    # Extract features and target
    features = ['city_encoded', 'day_of_week', 'day_of_year', 'hour', "city_latitude", "city_longitude",
                "day_feelslike", "day_snow", "day_snowdepth", "day_windgust", "day_winddir", "day_pressure",
                "day_precipprob", "day_severerisk", 'is_rus_holiday', 'is_ukr_holiday', 'vector_mean']
    X = data[features]
    y = data['alarm_occurred']

    # Split data into training and testing sets using TimeSeriesSplit
    tscv = StratifiedShuffleSplit(n_splits=10)

    best_accuracy = 0

    for j, (train_index, test_index) in enumerate(tscv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Naive Bayes model
        pipe = ImbPipeline([
            ('sampling', SMOTE()),
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])

        # Define hyperparameters to search over
        param_grid = {
            'sampling__k_neighbors': [3, 5, 7],
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }

        # Perform grid search using cross-validation RandomizedSearchCV?
        clf = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)

        result = permutation_importance(clf.best_estimator_, X_train, y_train, n_repeats=10, random_state=42)

        # Print feature importances
        for i, (importance, feature) in enumerate(zip(result.importances_mean, features)):
            print(f"Feature {i}: {feature} - Importance: {importance}")

        # Print the best hyperparameters and their corresponding accuracy
        print("Best hyperparameters:", clf.best_params_)
        print("accuracy:", clf.best_score_)

        # Check accuracy on the test set using the best model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Test set accuracy:", accuracy_score(y_test, y_pred))
        print("Test classification_report:\n", classification_report(y_pred, y_test))
        c_matrix = confusion_matrix(y_test, y_pred)
        print("Test confusion_matrix:\n", c_matrix)

        tn, fp, fn, tp = c_matrix.ravel()

        print("Confusion matrix:")
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Positives:", tp)

        print(j)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            filename = 'naive_bayes_v2.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(clf, f)


train()
