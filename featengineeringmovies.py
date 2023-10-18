import pandas as pd
import numpy as np

# Read in the data
df = pd.read_csv("C:\\Users\\Esther Huynh\\PycharmProjects\\tmdb_movies.csv")

# Exploring the data
print(df.info, "\n")
print(df.isnull().sum(), "\n")
print(df.shape)
print("\n\n")

# checking null types
print(df['homepage'].dtype)
print(df['overview'].dtype)
print(df['release_date'].dtype)
print(df['runtime'].dtype)
print(df['tagline'].dtype)
print("\n\n")

# Filling in the null numeric values
df['runtime'].fillna(df.runtime.interpolate(), inplace=True)

# Decomposing the date
df['release_date'] = pd.to_datetime(df['release_date'])
df['Year'] = df['release_date'].dt.year
df['Month'] = df['release_date'].dt.month.astype(str)
df['Day'] = df['release_date'].dt.day.astype(str)

print(df[['Year', 'Month', 'Day']].head())
print("\n\n")

# Adjusting the budget
df['log_budget'] = np.log(df['budget'] + 1)

# Encode inflation
df['inflation_budget'] = df['budget'] * (1 + (1.8 / 100) * (2018 - df['Year']))

# Ratios
df['budget_runtime_ratio'] = df['budget'] / df['runtime']
df['budget_popularity_ratio'] = df['budget'] / df['popularity']
df['budget_year_ratio'] = df['budget'] / (df['Year'] * df['Year'])
df['releaseYear_popularity_ratio'] = df['Year'] / df['popularity']

# Indicator Variables
# Has a homepage
df['has_homepage'] = 1
df.loc[pd.isnull(df['homepage']), "has_homepage"] = 0

# Was in English
df['is_English'] = np.where(df['original_language'] == 'en', 1, 0)

# Filling in missing categorical values
df = df.apply(lambda homepage: homepage.fillna(homepage.value_counts().index[0]))
df = df.apply(lambda overview: overview.fillna(overview.value_counts().index[0]))
df = df.apply(lambda release_date: release_date.fillna(release_date.value_counts().index[0]))
df = df.apply(lambda tagline: tagline.fillna(tagline.value_counts().index[0]))

# including only wanted variables
engineered_df = df[['budget_runtime_ratio', 'budget_popularity_ratio', 'budget_year_ratio',
                    'releaseYear_popularity_ratio', 'inflation_budget', 'Year', 'Month', 'is_English', 'has_homepage',
                    'budget', 'popularity', 'runtime', 'revenue']]

# One-hot encode categorical columns
engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
engineered_df = pd.concat([engineered_df, pd.get_dummies(engineered_df)], axis=1)

# Exploring new dataframe
print(engineered_df.isnull().sum(), "\n\n")
print(engineered_df.shape, "\n")

# Correlation Analysis
print(engineered_df.corr(), "\n\n")

# Training Test Split
train_engineered = engineered_df[['budget', 'runtime', 'popularity', 'has_homepage', 'budget_year_ratio',
                                  'is_English']].iloc[:3362]
train_baseline = engineered_df[['budget', 'runtime', 'popularity']].iloc[:3362]

test_engineered = engineered_df[['budget', 'runtime', 'popularity', 'has_homepage', 'budget_year_ratio',
                                 'is_English']].iloc[3362:]
test_baseline = engineered_df[['budget', 'runtime', 'popularity']].iloc[3362:]

target_train = engineered_df['revenue'].iloc[:3362]
target_test = engineered_df['revenue'].iloc[3362:]

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg_baseline = LinearRegression().fit(train_baseline, target_train)
reg_predict_baseline = reg_baseline.predict(test_baseline)

reg_engineered = LinearRegression().fit(train_engineered, target_train)
reg_predict_engineered = reg_engineered.predict(test_engineered)

rmse_baseline = np.sqrt(mean_squared_error(target_test, reg_predict_baseline))
rmse_engineered = np.sqrt(mean_squared_error(target_test, reg_predict_engineered))

rmse_difference = rmse_baseline - rmse_engineered

print("The difference in RMSE is", round(rmse_difference, 2), "dollars")

"""" Part B: 
What other questions can you ask about this dataset
    a. What predictions could you make?
        We could try predicting the popularity of the movie or the budget
    b. What features may be important for those predictions
        I think that some of the important features for predicting the popularity of the movie would be the 
        genre, the production companies, production countries, and the original language. Some of 
        the important features for predicting the budget are the genres, production company, production country, 
        popularity, and the inflation_budget ratio.
"""
# Using K-best method
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

engineered_df1 = df[['budget_runtime_ratio', 'budget_popularity_ratio', 'budget_year_ratio',
                     'releaseYear_popularity_ratio', 'inflation_budget', 'Year', 'Month', 'is_English', 'has_homepage',
                     'budget', 'popularity', 'runtime', 'revenue']]
# One Hot Encode
engineered_df1 = engineered_df1.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
engineered_df1 = pd.concat([engineered_df1, pd.get_dummies(engineered_df1)], axis=1)

feat = ['budget_popularity_ratio', 'budget_year_ratio',
        'releaseYear_popularity_ratio', 'inflation_budget', 'Year', 'Month', 'is_English', 'has_homepage',
        'budget', 'popularity', 'runtime']
label = 'revenue'

x, y = engineered_df1[feat].values, engineered_df1[label].values

print("Engineered Data Dimension: ", engineered_df1.shape)
print("Feature data dimension: ", x.shape)
print("Label data dimension: ", y.shape, "\n\n")

select = SelectKBest(score_func=chi2, k=5)
z = select.fit_transform(x, y)
print("After selecting best 5 features: ", z.shape)

print("All features:")
for i in range(0,11):
    print(feat[i], ", ", end='')

print("Selected best 5:")
print(feat[filter])
print(z)

