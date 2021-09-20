# perform our imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression

# read in our data
df = pd.read_csv('/Users/calebdill/Library/Mobile Documents/com~apple~CloudDocs/Syracuse/Data Analytics/Project/Combine Data.csv')

####### Data Cleansing ########

# view first five rows
df.head()

# view the data types of each column
df.dtypes

# count the number of rows missing data in each column
df.apply(lambda x: x.isnull().sum(), axis = 'rows')

## At first look, we should have enough data to just remove those with missing values as opposed to trying to fill them.
## However, after further analysis it appears the missing data is highly dependent on position.
## For example, it makes sense that QBs wouldn't participate in as many events at the combine, as that is common.
## How to handle these missing values? 2 options: replace with position median or replace with 0.
## For now, let's try the position median approach. 
## Assuming Team, Round, and Pick are all blank means the player did not get drafted. How to handle?
## Going to do 2 things: 
## 1. Round, pick, and team will be set as "undrafted" for the missing values
## 2. We will create an additional column which indicates whether or not a player went undrafted

# First, let's go ahead and convert round and pick to category types
df[['Round', 'Pick']] = df[['Round', 'Pick']].astype('category')
df.dtypes

# Replace missing values for Round, Team, and Pick with "undrafted"
df[['Team', 'Round', 'Pick']] = df[['Team', 'Round', 'Pick']].replace(np.NaN, 'Undrafted')

# view players which went undrafted
df.loc[df['Team'] == 'Undrafted']

# Add column indicating whether or not a player was drafted
df['Player Drafted'] = np.where(df['Team'] == 'Undrafted', 0, 1)

# Get median value for each position for our columns containing continuous data
dfPosMed = df.groupby('Pos').median()[['Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']]

# for anything we still don't have a value for, replace with 0
dfPosMed = dfPosMed.replace(np.NaN, 0)

# rename columns and join them into our original data set
dfPosMed = dfPosMed.rename(columns = {'Forty' : 'Median Forty', 'Vertical' : 'Median Vertical', 
'BenchReps' : 'Median BenchReps', 'BroadJump': 'Median BroadJump','Cone' : 'Median Cone', 'Shuttle' : 'Median Shuttle'})

dfPosMed = dfPosMed.reset_index()

df = df.merge(dfPosMed, on = 'Pos', how = 'left')

# replace missing values with median
df['Forty'] = df['Forty'].replace(np.NaN, df['Median Forty'])
df['Vertical'] = df['Vertical'].replace(np.NaN, df['Median Vertical'])
df['BenchReps'] = df['BenchReps'].replace(np.NaN, df['Median BenchReps'])
df['BroadJump'] = df['BroadJump'].replace(np.NaN, df['Median BroadJump'])
df['Cone'] = df['Cone'].replace(np.NaN, df['Median Cone'])
df['Shuttle'] = df['Shuttle'].replace(np.NaN, df['Median Shuttle'])

# check for other nulls
df.apply(lambda x: x.isnull().sum(), axis = 'rows')

# explore null Pfr_IDs
df.loc[df['Pfr_ID'].isnull()]
# Can't drop null Pfr_IDs since this would drop our Undrafted players and skew the data
# this is just an ID, so we can remove the column altogether and just create an index for each player

# create index
df = df.reset_index()

# drop Pfr_ID column
df = df.drop('Pfr_ID', axis = 1)
df.head()

## Good news - we managed to keep all of our original data without dropping records

####### EDA #########

# show the number of players drafted vs undrafted
plt.figure(figsize=(8,6))
sns.countplot(x = 'Player Drafted', data = df).set_title('Players Drafted Count')
## looks like most players got drafted. A little unbalanced,  but not terrible

# let's look at the number of players drafted by round 
plt.figure(figsize=(8,6))
sns.countplot(x = 'Round', data = df).set_title('Players Drafted by Round')
## undrafted is the leader here, which makes sense
## everything else is about even, wiith rounds 3 and 4 being the most common for drafted players

# finally, counts by position
plt.figure(figsize=(8,6))
sns.countplot(x = 'Pos', data = df).set_title('Count of each Pos')
## interesting insights here - a lot of CBs and WRs as part of our data set, regardless of draft position. 
## Let's dig deeper and look at draft rate per position

# groupby position and get the count of each position
dfPosCount = df.groupby('Pos').count()
dfPosCount = dfPosCount.reset_index()
dfPosCount = dfPosCount[['Pos', 'Player']]
dfPosCount = dfPosCount.rename(columns = {'Player': 'Position Count'})
dfPosCount

### An argument could be made that we need to consolidate some of these positions
### For example, SS and FS are both Safeties, while DB could be a catch all for Safeties and CBs
### For the purposes of this analysis, we will not make any changes to the positions
### It is possible a more "general" position impacts draft grade as opposed to a more "specific" position
### so let's see how everything plays out with the original data

# groupby position and get the number of drafted players at each position
dfPosDrafted = df.groupby('Pos').sum()['Player Drafted']
dfPosDrafted = dfPosDrafted.reset_index()
dfPosDrafted = dfPosDrafted.rename(columns = {'Player Drafted': 'Players Drafted'})
dfPosDrafted

dfPos = dfPosCount.merge(dfPosDrafted, on = 'Pos', how = 'left')
dfPos['Draft %'] = dfPos['Players Drafted'] / dfPos['Position Count'] * 100
dfPos['Draft %'] = round(dfPos['Draft %'])
dfPos = dfPos.sort_values(by = 'Draft %', ascending = False)
dfPos

# visualize draft %
plt.figure(figsize=(8,6))
sns.barplot(y = 'Draft %', x = 'Pos', data = dfPos).set_title('Draft % by Pos')

## Looks like OLB had the highest draft % with a few positions having no players drafted at all


#### Model Building ####
## Our goal is going to be to pick whether or not a player is drafted
## Going to remove columns which may mess up our model, such as "Pick" and "Round"
## These columns are being removed because they are not something we could actually use to predict our target variable
## If we know which pick someone was drafted in, then there would be no need to predict the round!

# remove "Pick", "Round", "AV", and "Team" columns
df = df.drop(['Pick', 'Round', 'AV', 'Team'], axis = 1)
df.head()

# remove Median columns as they are no longer needed
dfNew = df.drop(['Median Forty', 'Median Vertical', 'Median BenchReps', 'Median BroadJump',
'Median Cone', 'Median Shuttle'], axis = 1)
dfNew.head()

# convert year and index to type object
dfNew[['Year', 'index']] = dfNew[['Year', 'index']].astype(object)
dfNew.dtypes

### Model 1: Random Forest ###

## split our label and features
labels = np.array(dfNew['Player Drafted'])
labels
# dropping index and Player as well 
dfFeatures = dfNew.drop(['index', 'Player', 'Player Drafted', 'Year'], axis = 1)
dfFeatures.head()

# convert Pos to data type category
dfFeatures[['Pos']] = dfFeatures[['Pos']].astype(object)
dfFeatures.dtypes

# One-hot encode our categorical variables
dfDummy = pd.get_dummies(dfFeatures)
dfDummy.head()

# view all of our columns
dfDummy.columns

# save feature names
featureNames = list(dfDummy.columns)

# convert features to np array
features = np.array(dfDummy)
features.dtype


# split data into training and test sets
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
    features, labels, test_size = 0.25, random_state = 42
)

# define our random forest model
rfModel = RandomForestClassifier(n_estimators = 100, random_state = 42,
max_features = 'sqrt', n_jobs = -1, verbose = 1)

# train model
rfModel.fit(trainFeatures, trainLabels)

# get predictions with training data
rfPredictionsTrain = rfModel.predict(trainFeatures)

# get probabilities with training data
rfProbsTrain = rfModel.predict_proba(trainFeatures)

# get predictions with test data
rfPredictionsTest = rfModel.predict(testFeatures)

# get probabilities with test data
rfProbsTest = rfModel.predict_proba(testFeatures)

# create confusion matrix for test data
confMat = confusion_matrix(testLabels, rfPredictionsTest)
print(confMat)
sns.heatmap(confMat, annot = True)

# how did we do?
recall_score(testLabels, rfPredictionsTest)
precision_score(testLabels, rfPredictionsTest)
## 83.08% recall
## 69.5% precision

# compare with our training scores
recall_score(trainLabels, rfPredictionsTrain)
precision_score(trainLabels, rfPredictionsTrain)
## Model overfit?


# which features are most important?
featImportance = pd.DataFrame({'Feature': featureNames,
'Importance': rfModel.feature_importances_}).sort_values('Importance', ascending = False)
featImportance
### importance is greater for the actual drills and not position - interesting!

## Model 2: Logistic Regression ##

# define logReg model
lrModel = LogisticRegression()

# fit the model 
lrModel.fit(trainFeatures, trainLabels)

# get predictions with test data
lrPredictionsTest = lrModel.predict(testFeatures)

# get probabilities with test data
lrProbsTest = lrModel.predict_proba(testFeatures)

# create confusion matrix for test data
confMat = confusion_matrix(testLabels, lrPredictionsTest)
print(confMat)
sns.heatmap(confMat, annot = True)

# overall performance
print('Accuracy:', recall_score(testLabels, lrPredictionsTest))
print('Recall:', recall_score(testLabels, lrPredictionsTest))
print('Precision:', precision_score(testLabels, lrPredictionsTest))

## Performs a tiny bit better than our rf model

# feature importance
lrFeatImp = pd.DataFrame({'Feature': featureNames,
'Coefficient': [i for i in lrModel.coef_[0]]}).sort_values('Coefficient', ascending = False)
# interesting - a little different than our rf model!

# visualize coefficients
plt.figure(figsize=(8,6))
sns.barplot(y = 'Feature', x = 'Coefficient', data = lrFeatImp).set_title('Logistic Regression Model Featue Coefficients')
