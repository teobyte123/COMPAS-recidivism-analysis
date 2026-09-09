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