import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = "dir/alarms.csv"
df = pd.read_csv(data, sep=';')
df['times'] = df['id'].apply(lambda x: 1)
df['hour'] = df['start'].apply(lambda x: int(str(x).split()[1][:-6]))

emp = pd.DataFrame({'Alarms amount': df['times'],
                    'Hour': df['hour']
                    }, columns=['Alarms amount', 'Hour']).sort_values(by='Hour')

# PLOTTING
fig, ax = plt.subplots(figsize=(12, 9))
fig = sns.barplot(x="Hour", y="Alarms amount", data=emp,
                  estimator=sum, errorbar=None, ax=ax)

fig.axes.set_title("Alarms Starting Hours", fontsize=40)
fig.set_xlabel("Hour", fontsize=20)
fig.set_ylabel("Alarms amount", fontsize=20)

x_dates = emp['Hour'].unique()
ax.set_xticklabels(labels=x_dates, rotation=0, ha='center')

plt.show()
