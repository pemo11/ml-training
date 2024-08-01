# File: RandomForestClassifier.py
# Description: Random Forest Classifier model for Breast Cancer Diagnosis

# get the csv data from my github repo
import pandas as pd
csvPath = "https://raw.githubusercontent.com/pemo11/ml-training/main/Python/Breastcancer-Diagnosis/breast-cancerdata.csv"

df = pd.read_csv(csvPath)

# Display the data
print(df.info)
print(df.describe())
print(df.shape)

# Show me the types
df.dtypes

# Drop columns
df.drop(["Unnamed: 32","id"], axis=1, inplace=True)

# some kind of encoding
df.diagnosis=[1 if each=="M" else 0 for each in df.diagnosis]
print(df.head())

# Split training and test data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size = 0.2, random_state = 122)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
classifier.fit(X_train,y_train)

# Predict
train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_predictions)
print(cm)

### Train data accuracy
from sklearn.metrics import accuracy_score,f1_score

print("TRAIN Conf Matrix : \n", confusion_matrix(y_train, train_predictions))
print("\nTRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=0))

### Test data accuracy
print("\n\n--------------------------------------\n\n")

print("TEST Conf Matrix : \n", confusion_matrix(y_test, test_predictions))
print("\nTEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=0))