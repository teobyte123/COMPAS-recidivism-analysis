import pandas as pd

url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df = pd.read_csv(url)
df.to_csv("data/compas-scores-two-years.csv", index=False) #local path for raw dataset
print(df.shape)

#Phase 1 starts.
print(df.info())       #column types, non-null included
print(df.head(10))         #display first 10 rows
print(df.describe())       #summary stats for numeric columns
print(df['race'].value_counts())  #count of each race category
print(df['score_text'].value_counts())      # Low-Medium-High COMPAS risk scale
print(df['two_year_recid'].value_counts())  # variable: did they reoffend within 2 years?
 #cross-tabulation of race and COMPAS risk score
print(pd.crosstab(df['race'], df['score_text'], normalize='index'))     

#Phase 2 starts.
print(df['age'].describe())         #summary stats for age
print(df['priors_count'].describe())        #number of prior offenses

import matplotlib.pyplot as plt

#Histogram of priors offenses
df['priors_count'].hist(bins=30)
plt.xlabel('Number of prior offenses')
plt.ylabel('Count')
plt.title('Distribution of Prior Offenses')
plt.show()

#Cross-tabulation of COMPAS risk score and two-year recidivism
print(pd.crosstab(df['score_text'], df['two_year_recid'], normalize='index'))

#Include race in the cross-tabulation(African-American & Caucasian).
#Limited scope to African-American and Caucasian races because of significant sample size difference with other races.
for race in ['African-American', 'Caucasian']:
    subset = df[df['race'] == race]
    print(f"\n{race}:")
    print(pd.crosstab(subset['score_text'], subset['two_year_recid'], normalize='index'))

import seaborn as sns

#boxplot visualization of age distribution by COMPAS risk score
sns.boxplot(x='score_text', y='age', data=df, order=['Low', 'Medium', 'High'])
plt.title('Age Distribution by COMPAS Risk Score')
plt.show()

for race in ['African-American', 'Caucasian']:
    subset = df[df['race'] == race]
    print(f"\n{race}:")
    print(pd.crosstab(subset['two_year_recid'], subset['score_text'], normalize='index'))

df.groupby('race')['two_year_recid'].mean()         #average recidivism rate by race

#Phase 3 starts.
#Data cleaning and filtering based on specific criteria
df_clean = df[
    (df['days_b_screening_arrest'] <= 30) &         #screening within 30 days before arrest
    (df['days_b_screening_arrest'] >= -30) &        #screening within 30 days after arrest
    (df['is_recid'] != -1) &        #missing data for recidivism
    (df['c_charge_degree'] != 'O')        #no meaningful data for recidivism
].copy()

print(f"Original: {len(df)} rows -> After filtering: {len(df_clean)} rows")

features = ['sex', 'age', 'race', 'priors_count', 'c_charge_degree']        #inputs/features searched for by the model
target = 'two_year_recid'       #what's attempting to be predicted by the model

df_model = df_clean[features + [target]].copy()
df_model.head()

#Categorical variables
df_model['sex'] = df_model['sex'].map({'Male': 0, 'Female': 1})
df_model['c_charge_degree'] = df_model['c_charge_degree'].map({'F': 0, 'M': 1})

df_model = pd.get_dummies(df_model, columns=['race'], drop_first=False)

df_model.head()
df_model.dtypes
df_model.isnull().sum()

#Phase 4 starts.
#Baseline model using logistic regression to predict two-year recidivism
from sklearn.model_selection import train_test_split

X = df_model.drop(columns=['two_year_recid'])
y = df_model['two_year_recid']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Inspecting coefficients; to see what the model has learned.
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', ascending=False)

print(coefficients)

X_train_ref = X_train.drop(columns=['race_Caucasian'])
X_test_ref = X_test.drop(columns=['race_Caucasian'])

model_ref = LogisticRegression(max_iter=1000)
model_ref.fit(X_train_ref, y_train)

coefficients_ref = pd.DataFrame({
    'feature': X_train_ref.columns,
    'coefficient': model_ref.coef_[0]
}).sort_values('coefficient', ascending=False)

print(coefficients_ref)
print(f"Test accuracy: {model_ref.score(X_test_ref, y_test):.3f}")

df_clean.groupby('race')['priors_count'].mean()

