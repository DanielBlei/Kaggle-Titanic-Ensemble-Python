import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.svm import SVC

Tloc = "/home/danielblei/PycharmProjects/TitanicKaggle/train.csv"
TstLoc = "/home/danielblei/PycharmProjects/TitanicKaggle/test.csv"

trainM = pd.read_csv(Tloc)
trainG = pd.read_csv(Tloc)
testP = pd.read_csv(TstLoc)
testSub = testP['PassengerId']

# trainM.shape
# trainM.describe()

def boxplot ():
    trainG.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False)
    plt.show()

def histogram ():
    trainG.hist()
    plt.show()

def corr():
    corrr = trainG.corr()
    sns.heatmap(corrr,cmap="YlGnBu",annot=True,
                xticklabels=cor.columns,
                yticklabels=cor.columns)

#Pandas warming off:
pd.options.mode.chained_assignment = None

# Feature Engineer:

trainM['Title'] = trainM.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#print (pd.crosstab(trainM['Title'], trainM['Sex']))

trainM['Title'] = trainM['Title'].replace(['Lady', 'Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
trainM['Title'] = trainM['Title'].replace(['Mlle'],'Miss')
trainM['Title'] = trainM['Title'].replace(['Ms'], 'Miss')
trainM['Title'] = trainM['Title'].replace(['Mme'], 'Mrs')
print ("")
print (trainM[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

testP['Title'] = testP.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

testP['Title'] = testP['Title'].replace(['Lady', 'Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
testP['Title'] = testP['Title'].replace(['Mlle'],'Miss')
testP['Title'] = testP['Title'].replace(['Ms'], 'Miss')
testP['Title'] = testP['Title'].replace(['Mme'], 'Mrs')

#Looking for missing values (NaN):
# trainM.Age.isnull().values.sum()
# trainP.....isnull()...


#Converting and filling missing values:

title_fill = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
trainM['Title'] = trainM['Title'].map(title_fill)
trainM['Title'] = trainM['Title'].fillna(0)
testP['Title'] = testP['Title'].map(title_fill)
testP['Title'] = testP['Title'].fillna(0)

trainM['Sex'] = trainM['Sex'].replace('male', 0)
trainM['Sex'] = trainM['Sex'].replace('female', 1)
testP['Sex'] = testP['Sex'].replace('male', 0)
testP['Sex'] = testP['Sex'].replace('female', 1)

trainM['Age'].fillna((trainM['Age'].mean()), inplace=True)
testP['Age'].fillna((testP['Age'].mean()), inplace=True)
trainM['Age'] = trainM['Age'].astype(int)
testP['Age'] = testP['Age'].astype(int)

#Checking: trainM.Age.isnull().values.sum() = 0

#splitting Age:

trainM['AgeBand'] = pd.cut(trainM['Age'], 5)
print ("")
print(trainM[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

trainM.Age[trainM['Age'] <= 16] = 0
trainM.Age[(trainM['Age'] > 16) & (trainM['Age'] <= 32)] = 1
trainM.Age[(trainM['Age'] > 32) & (trainM['Age'] <= 48)] = 2
trainM.Age[(trainM['Age'] > 48) & (trainM['Age'] <= 64)] = 3
trainM.Age[trainM['Age'] > 64] = 4

testP.Age[testP['Age'] <= 16] = 0
testP.Age[(testP['Age'] > 16) & (testP['Age'] <= 32)] = 1
testP.Age[(testP['Age'] > 32) & (testP['Age'] <= 48)] = 2
testP.Age[(testP['Age'] > 48) & (testP['Age'] <= 64)] = 3
testP.Age[ testP['Age'] > 64] = 4

#checking family dependencies:

trainM['FamilySize'] = trainM['SibSp'] + trainM['Parch'] + 1
testP['FamilySize'] = trainM['SibSp'] + trainM['Parch'] + 1
print('')
print(trainM[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Shkring the Family Size variable by the surviving rate:

trainM.FamilySize[trainM['FamilySize'] <= 4] = 1
trainM.FamilySize[(trainM['FamilySize'] > 4) & (trainM['FamilySize'] <= 7)] = 2
trainM.FamilySize[trainM['FamilySize'] > 7] = 3

testP.FamilySize[testP['FamilySize'] <= 4] = 1
testP.FamilySize[(testP['FamilySize'] > 4) & (testP['FamilySize'] <= 7)] = 2
testP.FamilySize[testP['FamilySize'] > 7] = 3

#Lonely travels variable: Did not bring any considerable improve.

# trainM["Alone"] = np.where(trainM['SibSp'] + trainM['Parch'] + 1 == 1, 1,0)
# testP["Alone"] = np.where(testP['SibSp'] + testP['Parch'] + 1 == 1, 1,0)

#Converting Embarked variable and filling NaNs:

EmbarkM = trainM.Embarked.dropna().mode()[0]

trainM['Embarked'] = trainM['Embarked'].fillna(EmbarkM)

print('')
print(trainM[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

trainM['Embarked'] = trainM['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
testP['Embarked'] = testP['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)

#Converting Cabin variable and filling NaNs: Did not bring any considerable improve.

# trainM["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in trainM['Cabin']])
#
# testP["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in testP['Cabin']])
# #trainM["Cabin"].unique()
#
# trainM['Cabin'] = trainM['Cabin'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,'F': 6, 'G': 7, 'T': 8, 'X': 0}).astype(int)
# testP['Cabin'] = testP['Cabin'].map({'A': 1, 'B': 2, 'C': 3,'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'X': 0}).astype(int)
#
# print('')
# print(trainM[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))


#Splitting Fare attribute

testP['Fare'].fillna(testP['Fare'].dropna().median(), inplace=True)

trainM['FareBand'] = pd.qcut(trainM['Fare'], 6)
print('')
print(trainM[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

trainM.Fare[trainM['Fare'] <= 8] = 0
trainM.Fare[(trainM['Fare'] > 8) & (trainM['Fare'] <= 14)] = 1
trainM.Fare[(trainM['Fare'] > 14) & (trainM['Fare'] <= 26)] = 2
trainM.Fare[(trainM['Fare'] > 26) & (trainM['Fare'] <= 52)] = 3
trainM.Fare[trainM['Fare'] > 52 & (trainM['Fare'] <= 70)] = 4
trainM.Fare[trainM['Fare'] > 70 & (trainM['Fare'] <= 100)] = 4
trainM.Fare[trainM['Fare'] > 100] = 5

testP.Fare[testP['Fare'] <= 8] = 0
testP.Fare[(testP['Fare'] > 8) & (testP['Fare'] <= 14)] = 1
testP.Fare[(testP['Fare'] > 14) & (testP['Fare'] <= 26)] = 2
testP.Fare[(testP['Fare'] > 26) & (testP['Fare'] <= 52)] = 3
testP.Fare[testP['Fare'] > 52 & (testP['Fare'] <= 70)] = 4
testP.Fare[testP['Fare'] > 70 & (testP['Fare'] <= 100)] = 4
testP.Fare[testP['Fare'] > 100] = 5

trainM['Fare'] = trainM['Fare'].astype(int)
testP['Fare'] = testP['Fare'].astype(int)


#Dropping variables:

trainM = trainM.drop(['Ticket','AgeBand','Cabin', 'FareBand','Name', 'PassengerId','Parch', 'SibSp'], axis=1)
testP = testP.drop(['Ticket','Name','Cabin', 'PassengerId','Parch', 'SibSp'], axis=1)

X_train = np.array((trainM.drop("Survived", axis=1)))
Y_train = np.array(trainM["Survived"])

kfold = StratifiedKFold(n_splits=7)

#Parameter Tuning:

#SVM:
print("")

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.01, 0.1, 0.18],
                  'C': [1,165,175,185]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVM_best = gsSVMC.best_estimator_

print ('SVM best accuracy score:', gsSVMC.best_score_)

#RandomForest:
print("")

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [6,3],
              "min_samples_split": [2, 3],
              "min_samples_leaf": [1, 2],
              "bootstrap": [False],
              "n_estimators" :[433,450],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RF_best = gsRFC.best_estimator_

print ('RandomForest best accuracy score:',gsRFC.best_score_)

#XGboost
print("")
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["exponential"],
              'n_estimators' : [10,433,450],
              'learning_rate': [0.18],
              'max_depth': [8],
              'min_samples_leaf': [10,15,20],
              'max_features': [0.3],
              'subsample' : [0.8]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

XGB_best = gsGBC.best_estimator_

print ('XGboost best accuracy score:', gsGBC.best_score_)


votingC = VotingClassifier(estimators=[('rfc', RF_best),
('svc', SVM_best),('gbc',XGB_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

preds = votingC.predict(X_train)

print('')
print("Voting Classifier Accuracy Score:")
print(accuracy_score(Y_train, preds))
print("F-score:")
print(fbeta_score(Y_train, preds, beta=0.5))

test_Survived = pd.Series(votingC.predict(testP), name="Survived")

results = pd.concat([testSub,test_Survived],axis=1)

results.to_csv("ensemble_voting.csv",index=False)
