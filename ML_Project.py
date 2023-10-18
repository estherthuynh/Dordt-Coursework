import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# The file was in Excel
# I used the Excel file to insert a row for the feature names and changed the numbers in the class and specimen cols
# to their names

# Classification question: Can we create an algorithm that can predict the class of the leaf based on the other feats?
# Since we already know the class for each leaf -> supervised

# Read in the file
leaf = pd.read_excel("C:\\Users\\Esther Huynh\\PycharmProjects\\leaf.xlsx")
print()

# Exploring the data
print(leaf.head(), "\n")
print(leaf.shape)
print(leaf.columns, "\n")
print(leaf.isnull().sum(), "\n")

# The data is clean (no nulls)
##

# Get dummies for specimen type (simple vs complex)
Specimen_Type = pd.get_dummies(leaf.Specimen_En)
leaf = pd.concat([leaf, Specimen_Type], axis=1)

# Reset the data
leaf = pd.read_excel("C:\\Users\\Esther Huynh\\PycharmProjects\\leaf.xlsx")
leaf = pd.get_dummies(leaf, columns=['Specimen_En'])

# Since I changed the specimen to simple or complex (see the documentation on pdf), I don't need the specimen #
# Drop Specimen_Num column
leaf.drop('Specimen_Num', axis=1, inplace=True)
# Drop mutually exclusive columns
leaf.drop('Specimen_En_Complex', axis=1, inplace=True)

print("New Columns:")
print(leaf.columns, "\n\n")

# Find the prediction accuracy using RandomForest, GradientBoost, & Bagging Classifiers on orig. data for comparison

feat = ['Eccentricity', 'Aspect_Ratio', 'Elongation', 'Solidity', 'Stochastic_Convexity',
        'Isoperimetric_Factor', 'Maximal_Indentation_Depth', 'Lobedness', 'Average_Intensity', 'Average_Contrast',
        'Smoothness', 'Third_moment', 'Uniformity', 'Entropy', 'Specimen_En_Simple']
label = 'Class'

X, y = leaf[feat].values, leaf[label].values
print(leaf.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,)

## Random Forest
random_clf_OG = RandomForestClassifier(n_estimators=10)
random_clf_OG.fit(X_train, y_train)
randomF_pred_OG = random_clf_OG.predict(X_test)
print("Random Forest Classifier Accuracy on Original Dataframe:")
print(accuracy_score(y_test, randomF_pred_OG), "\n")

## Gradient Booster
gradient_clf_OG = GradientBoostingClassifier(n_estimators=10)
gradient_clf_OG.fit(X_train, y_train)
gradientB_pred_OG = gradient_clf_OG.predict(X_test)
print("GradientBooster Classifier Accuracy on Original Dataframe:")
print(accuracy_score(y_test, gradientB_pred_OG), "\n")

## Bagging
pipeline_OG = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, max_iter=1000))
bagging_clf_OG = BaggingClassifier(base_estimator=pipeline_OG, n_estimators=10)
bagging_clf_OG.fit(X_train, y_train)
baggingC_pred_OG = bagging_clf_OG.predict(X_test)
print("Bagging Classifier Accuracy on Original Dataframe:")
print(accuracy_score(y_test, baggingC_pred_OG), "\n\n")


# Feature Engineering using K-best Method to find top 5 features
from sklearn.feature_selection import SelectKBest, chi2

select = SelectKBest(score_func=chi2, k=5)
select.fit_transform(X, y)
Top_Five = list(leaf.columns[select.get_support(indices=True)])
print("Top Five Features Using the K-Best Method:")
for i in range(0, 5):
    print(Top_Five[i], ", ", sep='', end='')

print("\n\n")

# Create new dataframe with only wanted variables
engineered_leaf = leaf[['Class', 'Eccentricity', 'Aspect_Ratio', 'Stochastic_Convexity', 'Maximal_Indentation_Depth',
                        'Uniformity']]

# Using same Classifier methods on the engineered Dataframe
engineered_feat = ['Eccentricity', 'Aspect_Ratio', 'Stochastic_Convexity', 'Maximal_Indentation_Depth', 'Uniformity']
engineered_label = 'Class'

X_New, y_New = engineered_leaf[engineered_feat].values, engineered_leaf[engineered_label].values
# Train-test split
X_New_train, X_New_test, y_New_train, y_New_test = train_test_split(X_New, y_New,)


## Random Forest
random_clf_New = RandomForestClassifier(n_estimators=10)
random_clf_New.fit(X_New_train, y_New_train)
randomF_pred_New = random_clf_New.predict(X_New_test)
print("Random Forest Classifier Accuracy on Engineered Dataframe:")
print(accuracy_score(y_New_test, randomF_pred_New), "\n")

## Gradient Booster
gradient_clf_New = GradientBoostingClassifier(n_estimators=10)
gradient_clf_New.fit(X_New_train, y_New_train)
gradientB_pred_New = gradient_clf_New.predict(X_New_test)
print("GradientBooster Classifier Accuracy on Engineered Dataframe:")
print(accuracy_score(y_New_test, gradientB_pred_New), "\n")

## Bagging
pipeline_New = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, max_iter=1000))
pipeline_New.fit(X_New_train, y_New_train)
bagging_clf_New = BaggingClassifier(base_estimator=pipeline_New, n_estimators=10)
bagging_clf_New.fit(X_New_train, y_New_train)
baggingC_pred_New = bagging_clf_New.predict(X_New_test)
print("Bagging Classifier Accuracy on Engineered Dataframe:")
print(accuracy_score(y_New_test, baggingC_pred_New), "\n\n")



###########
## Result Analysis
print("Result Analysis:")
print("I was able to create 2 sets of 3 separate models that were able to predict the class of the leaves given the \n"
      "features. The first set of models used all the features to make the predictions and returned slightly more \n"
      "accurate predictions than the second set of 3. For the second set, I used feature selection (K-Best Method) \n"
      "to return the best 5 features of the original dataframe. Using these 5 features, I created a new dataframe with \n"
      "only these features and ran the same models on this dataset. The models were still able to make predictions \n"
      "using this dataset, but had a slightly lower prediction accuracy than the first set of models. I found this \n"
      "interesting because the second model was engineered to use the best features, so I thought it would have a \n"
      "higher accuracy than the first set. I think using this method with larger datasets where overfitting might be \n"
      "an issue might have better results for the second set because the feature engineering/selection would \n"
      "help cut down the data points that are skewing the results.")
