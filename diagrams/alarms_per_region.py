import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = "dir/alarms.csv"
df = pd.read_csv(data, sep=';')
df['times'] = df['id'].apply(lambda x: 1)

emp = pd.DataFrame({'Alarms amount': df['times'],
                    'Region': df['region_title']
                    }, columns=['Alarms amount', 'Region'])

# PLOTTING
fig, ax = plt.subplots(figsize=(15, 15))
fig = sns.barplot(x="Region", y="Alarms amount", data=emp,
                  estimator=sum, errorbar=None, ax=ax)

fig.axes.set_title("Alarms per region", fontsize=40)
fig.set_xlabel("Region", fontsize=20)
fig.set_ylabel("Alarms amount", fontsize=20)

x_regions = emp['Region'].unique()
ax.set_xticklabels(labels=x_regions, rotation=45, ha='right')

plt.show()
