import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedShuffleSplit
from utils import get_region_number


def train(data):
    # Preprocess the data
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
    tscv = StratifiedShuffleSplit(n_splits=2)

    best_accuracy = 0
    best_clf = 0

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

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf

    return best_clf
