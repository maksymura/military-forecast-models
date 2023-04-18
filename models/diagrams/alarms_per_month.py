import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = "dir/alarms.csv"
df = pd.read_csv(data, sep=';')
df['times'] = df['id'].apply(lambda x: 1)
df['date'] = df['start'].apply(lambda x: str(x).split()[0][:-2] + "01")

emp = pd.DataFrame({'Alarms amount': df['times'],
                    'Date month': pd.to_datetime(df['date'])
                    }, columns=['Alarms amount', 'Date month'])

# PLOTTING
fig, ax = plt.subplots(figsize=(12, 9))
fig = sns.barplot(x="Date month", y="Alarms amount", data=emp,
                  estimator=sum, errorbar=None, ax=ax)

fig.axes.set_title("Alarms per month", fontsize=40)
fig.set_xlabel("Date month", fontsize=20)
fig.set_ylabel("Alarms amount", fontsize=20)

x_dates = emp['Date month'].dt.strftime('%Y-%m').unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.show()
