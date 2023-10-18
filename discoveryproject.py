import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df = pd.read_excel("C:\\Users\\Esther Huynh\\PycharmProjects\\AssocRules_Dataset_Students.xlsx")
print()
print(df.isnull().sum(), "\n")
print(df.head(), "\n")

# clean the data
# drop transfer students
df.dropna(axis=0, subset=['FSStriving'], inplace=True)
# drop residual columns
df.drop('Predicted Residual', axis=1, inplace=True)
df.drop('First Sem Residual', axis=1, inplace=True)

print(df.columns, "\n")
print(df.shape)

# rounding gpa to 2 decimals
df['StrivingGPA'] = df['StrivingGPA'].round(decimals=1)
df['CumGPA'] = df['CumGPA'].round(decimals=1)
df['HSGPA'] = df['HSGPA'].round(decimals=1)
df['FSStriving'] = df['FSStriving'].round(decimals=1)
df['FirstSemGPA'] = df['FirstSemGPA'].round(decimals=1)

# changing GPA to categorical
for i in range(0, 3145):
    if df['StrivingGPA'].iloc[i] >= 3.5:
        df['StrivingGPA'].iloc[i] = 'A'
    elif 2.5 <= df['StrivingGPA'].iloc[i] < 3.5:
        df['StrivingGPA'].iloc[i] = 'B'
    elif 1.5 <= df['StrivingGPA'].iloc[i] < 2.5:
        df['StrivingGPA'].iloc[i] = 'C'
    elif 1.0 <= df['StrivingGPA'].iloc[i] < 1.5:
        df['StrivingGPA'].iloc[i] = 'D'
    else:
        df['StrivingGPA'].iloc[i] = 'F'

    if df['CumGPA'].iloc[i] >= 3.5:
        df['CumGPA'].iloc[i] = 'A'
    elif 2.5 <= df['CumGPA'].iloc[i] < 3.5:
        df['CumGPA'].iloc[i] = 'B'
    elif 1.5 <= df['CumGPA'].iloc[i] < 2.5:
        df['CumGPA'].iloc[i] = 'C'
    elif 1.0 <= df['CumGPA'].iloc[i] < 1.5:
        df['CumGPA'].iloc[i] = 'D'
    else:
        df['CumGPA'].iloc[i] = 'F'

    if df['HSGPA'].iloc[i] >= 3.5:
        df['HSGPA'].iloc[i] = 'A'
    elif 2.5 <= df['HSGPA'].iloc[i] < 3.5:
        df['HSGPA'].iloc[i] = 'B'
    elif 1.5 <= df['HSGPA'].iloc[i] < 2.5:
        df['HSGPA'].iloc[i] = 'C'
    elif 1.0 <= df['HSGPA'].iloc[i] < 1.5:
        df['HSGPA'].iloc[i] = 'D'
    else:
        df['HSGPA'].iloc[i] = 'F'

    if df['FSStriving'].iloc[i] >= 3.5:
        df['FSStriving'].iloc[i] = 'A'
    elif 2.5 <= df['FSStriving'].iloc[i] < 3.5:
        df['FSStriving'].iloc[i] = 'B'
    elif 1.5 <= df['FSStriving'].iloc[i] < 2.5:
        df['FSStriving'].iloc[i] = 'C'
    elif 1.0 <= df['FSStriving'].iloc[i] < 1.5:
        df['FSStriving'].iloc[i] = 'D'
    else:
        df['FSStriving'].iloc[i] = 'F'

    if df['FirstSemGPA'].iloc[i] >= 3.5:
        df['FirstSemGPA'].iloc[i] = 'A'
    elif 2.5 <= df['FirstSemGPA'].iloc[i] < 3.5:
        df['FirstSemGPA'].iloc[i] = 'B'
    elif 1.5 <= df['FirstSemGPA'].iloc[i] < 2.5:
        df['FirstSemGPA'].iloc[i] = 'C'
    elif 1.0 <= df['FirstSemGPA'].iloc[i] < 1.5:
        df['FirstSemGPA'].iloc[i] = 'D'
    else:
        df['FirstSemGPA'].iloc[i] = 'F'

# getting dummy variables for GPA
Striving_GPA = pd.get_dummies(df.StrivingGPA, prefix="Striving_GPA")
df = pd.concat([df, Striving_GPA], axis=1)
###
Cumm_GPA = pd.get_dummies(df.CumGPA, prefix="Cumm_GPA")
df = pd.concat([df, Cumm_GPA], axis=1)
###
HS_GPA = pd.get_dummies(df.HSGPA, prefix="HS_GPA")
df = pd.concat([df, HS_GPA], axis=1)
###
FS_Striving = pd.get_dummies(df.FSStriving, prefix="FS_Striving")
df = pd.concat([df, FS_Striving], axis=1)
###
First_Sem_GPA = pd.get_dummies(df.FirstSemGPA, prefix="First_Sem_GPA")
df = pd.concat([df, First_Sem_GPA], axis=1)

# get dummies for distance and delta
Distance = pd.get_dummies(df.Distance, prefix="Distance")
df = pd.concat([df, Distance], axis=1)
Predicted_Delta = pd.get_dummies(df.PredictedDelta, prefix="Predicted_Delta")
df = pd.concat([df, Predicted_Delta], axis=1)
FS_Delta = pd.get_dummies(df.FSDelta, prefix="FS_Delta")
df = pd.concat([df, FS_Delta], axis=1)

# drop mutually exclusive columns
df.drop('StrivingGPA', axis=1, inplace=True)
df.drop('CumGPA', axis=1, inplace=True)
df.drop('HSGPA', axis=1, inplace=True)
df.drop('FSStriving', axis=1, inplace=True)
df.drop('FirstSemGPA', axis=1, inplace=True)
df.drop('Striving_GPA_B', axis=1, inplace=True)
df.drop('Striving_GPA_C', axis=1, inplace=True)
df.drop('Striving_GPA_D', axis=1, inplace=True)
df.drop('Striving_GPA_F', axis=1, inplace=True)
df.drop('Cumm_GPA_B', axis=1, inplace=True)
df.drop('Cumm_GPA_C', axis=1, inplace=True)
df.drop('Cumm_GPA_D', axis=1, inplace=True)
df.drop('Cumm_GPA_F', axis=1, inplace=True)
df.drop('HS_GPA_B', axis=1, inplace=True)
df.drop('HS_GPA_C', axis=1, inplace=True)
df.drop('HS_GPA_F', axis=1, inplace=True)
df.drop('FS_Striving_B', axis=1, inplace=True)
df.drop('FS_Striving_C', axis=1, inplace=True)
df.drop('FS_Striving_D', axis=1, inplace=True)
df.drop('FS_Striving_F', axis=1, inplace=True)
df.drop('First_Sem_GPA_B', axis=1, inplace=True)
df.drop('First_Sem_GPA_C', axis=1, inplace=True)
df.drop('First_Sem_GPA_D', axis=1, inplace=True)
df.drop('First_Sem_GPA_F', axis=1, inplace=True)
df.drop('Distance', axis=1, inplace=True)
df.drop('Distance_1000M', axis=1, inplace=True)
df.drop('Distance_1001M', axis=1, inplace=True)
df.drop('Distance_500M', axis=1, inplace=True)
df.drop('PredictedDelta', axis=1, inplace=True)
df.drop('Predicted_Delta_Above', axis=1, inplace=True)
df.drop('Predicted_Delta_Below', axis=1, inplace=True)
df.drop('FSDelta', axis=1, inplace=True)
df.drop('FS_Delta_Above', axis=1, inplace=True)
df.drop('FS_Delta_Below', axis=1, inplace=True)

freq_items = apriori(df, min_support=0.4, use_colnames=True, verbose=1)
print(freq_items.head(7), "\n\n")

rules = association_rules(freq_items, metric="confidence", min_threshold=0.9)
print(rules.head(10), "\n\n")


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))
plt.show()
