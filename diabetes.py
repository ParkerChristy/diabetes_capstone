import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pickle


# import diabetes data
diabetes = pd.read_csv("data/diabetes.csv")
diabetes.head()


# drop pregnancies because men can't bear children
diabetes.drop(["BloodPressure"], axis=1, inplace=True)
# drop diabetes pedigree fn to simplify data entry
diabetes.drop(["DiabetesPedigreeFunction"], axis=1, inplace=True)

diabetes.head()


# change erroneous 0 values to null for a better count
# Pregnancies unchanged due to men's inability to bear children
diabetes[["Glucose","SkinThickness","Insulin","BMI"]] = diabetes[["Glucose","SkinThickness","Insulin","BMI"]].replace(0, np.NaN)


# check for / count null values
diabetes.isnull().sum()


# fill null values with median
diabetes.groupby(["Outcome"])["Glucose"].median()
diabetes["Glucose"].fillna(diabetes.groupby(["Outcome"])["Glucose"].transform("median"),inplace=True)


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


# show heatmap for visualization of correlation
plt.figure(figsize=(6,4))
sns.heatmap(diabetes.corr(),cmap='Reds',annot=False);


# show comparison distributions of each column vs. the outcome
for col in diabetes.columns:
    if col != "Outcome":
        sns.catplot("Outcome", col, data = diabetes)


# split into x/y
x = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2)


# build machine learning model
np.random.seed(42)

clf = RandomForestClassifier(n_estimators=500)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


clf.predict(x_test)


np.array(y_test)


# compare predictions to truth
y_preds = clf.predict(x_test)
np.mean(y_preds == y_test)


# save model to file
pickle.dump(clf, open("random_forest_model.pkl", "wb"))


# load model
loaded_pickle_model = pickle.load(open("random_forest_model.pkl", "rb"))