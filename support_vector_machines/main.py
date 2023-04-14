import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import TimeSeriesSplit


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
    return -1


# Preprocess data
data = "dir/final_data_new.csv"
df = pd.read_csv(data)

print('Process dataset')
df.drop(['day_description', 'city_latitude', 'city_longitude', 'city_address', 'day_stations', 'day_conditions'],
        axis=1)
df['has_alarm'] = df['alarm_start'].notna().astype(int)
df.drop(['alarm_start', 'alarm_end'], axis=1)

df['year'] = df['date'].apply(lambda x: str(x).split("-")[0])
df['month'] = df['date'].apply(lambda x: str(x).split("-")[1])
df['day'] = df['date'].apply(lambda x: str(x).split("-")[2])
df['hour'] = df['time'].apply(lambda x: str(x).split(":")[0])
df.drop(['date', 'time'], axis=1)

df['city_n'] = df['city'].apply(lambda x: get_region_number(str(x)))
df.drop(['city'], axis=1)

print('Processing vector started')
df['vector'] = df['vector'].apply(lambda x: [float(i) for i in str(x).strip('[]').split(',')])
print('Mean vector started')
df['vector'] = df['vector'].apply(np.mean)
print('Processing vector finished')

X = df[['year', 'month', 'day', 'hour', 'is_rus_holiday', 'is_ukr_holiday', 'city_n', 'day_feelslikemax',
        'day_feelslikemin', 'day_sunriseEpoch', 'hour_feelslike', 'day_feelslike', 'day_precipprob', 'day_snow',
        'day_snowdepth', 'day_windgust', 'day_windspeed', 'day_winddir', 'day_pressure', 'day_cloudcover',
        'day_visibility', 'day_severerisk', 'day_windgust', 'day_windspeed', 'day_winddir', 'day_cloudcover',
        'day_visibility', 'vector']]
y = df['has_alarm']

print('Model setting up')
svm_m = SVC(kernel='rbf', C=1, gamma='scale')
svm_bag = BaggingClassifier(svm_m, n_estimators=10, max_samples=0.5, max_features=0.5, n_jobs=-1)

confusion_matrices = []
accuracy_scores = []

print('Model setting up')
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print('Imputing missing values')
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    print('Missing values imputed')

    print('predict')
    svm_bag.fit(X_train_imputed, y_train)
    y_pred = svm_bag.predict(X_test_imputed)

    print('matrix')
    # Calculate the confusion matrix and append it to the list
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    print('accuracy')
    # Calculate the accuracy score and append it to the list
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)

avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
print("Average Confusion Matrix:\n", avg_confusion_matrix)
avg_accuracy_score = np.mean(accuracy_scores)
print("Average Accuracy Score:", avg_accuracy_score)

print("Paint Average Confusion Matrix")
tn, fp, fn, tp = avg_confusion_matrix.ravel()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['True Negatives', 'False Positives', 'False Negatives', 'True Positives'], [tn, fp, fn, tp])
ax.set_xlabel('Predicted label')
ax.set_ylabel('Number of instances')
ax.set_title('Confusion matrix')
plt.show()

# save the model as a pickle file
filename = 'svm.pickle'
with open(filename, 'wb') as f:
    pickle.dump(svm_bag, f)
