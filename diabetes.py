import numpy as np
import pandas as pd


# import diabetes data
diabetes = pd.read_csv("data/diabetes.csv")
diabetes.head()


# drop pregnancies because men can't bear children
diabetes.drop(["Pregnancies"], axis=1, inplace=True)
# drop diabetes pedigree fn to simplify data entry
diabetes.drop(["DiabetesPedigreeFunction"], axis=1, inplace=True)

diabetes.head()


# change erroneous 0 values to null for a better count 
diabetes[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = diabetes[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0, np.NaN)


# check for / count null values
diabetes.isnull().sum()


# fill null values with median
diabetes.groupby(["Outcome"])["Glucose"].median()
diabetes["Glucose"].fillna(diabetes.groupby(["Outcome"])["Glucose"].transform("median"),inplace=True)


# fill null values with median
diabetes.groupby(["Outcome"])["BloodPressure"].median()
diabetes["BloodPressure"].fillna(diabetes.groupby(["Outcome"])["BloodPressure"].transform("median"),inplace=True)


# fill null values with median
diabetes.groupby(["Outcome"])["SkinThickness"].median()
diabetes["SkinThickness"].fillna(diabetes.groupby(["Outcome"])["SkinThickness"].transform("median"),inplace=True)


# fill null values with median
diabetes.groupby(["Outcome"])["Insulin"].median()
diabetes["Insulin"].fillna(diabetes.groupby(["Outcome"])["Insulin"].transform("median"),inplace=True)


# fill null values with median
diabetes.groupby(["Outcome"])["BMI"].median()
diabetes["BMI"].fillna(diabetes.groupby(["Outcome"])["BMI"].transform("median"),inplace=True)


# check for / count null values after fill
diabetes.isnull().sum()


# histogram to visualize BMI distribution
diabetes["BMI"].plot.hist();


# histogram to visualize blood pressure distribution
diabetes["BloodPressure"].plot.hist();


# histogram to visualize insulin distribution
diabetes["Insulin"].plot.hist();


# histogram to visualize Glucose distribution
diabetes["Glucose"].plot.hist();


from sklearn.model_selection import train_test_split

# split into x/y
x = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2)


# build machine learning model
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

clf = RandomForestClassifier(n_estimators=500)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


clf.predict(x_test)


np.array(y_test)


# compare predictions to truth
y_preds = clf.predict(x_test)
np.mean(y_preds == y_test)


import pickle

# save model to file
pickle.dump(clf, open("random_forest_model.pkl", "wb"))


# load model
loaded_pickle_model = pickle.load(open("random_forest_model.pkl", "rb"))