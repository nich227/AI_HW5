'''
CS 6364.002

Homework 5
----------
Name:       Kevin Chen
NetID:      nkc160130
Instructor: Professor Chen
Due:        09/28/2020
'''
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Regression import *

# Q1
housing_df = pd.read_csv("HousingData.csv")
housing_df.head()

# Preprocessing data

# Impute the null values
simp_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
housing_df.replace('?', np.nan, inplace=True)
housing_df_columns = housing_df.columns
housing_df_index = housing_df.index
housing_df = pd.DataFrame(simp_imp.fit_transform(housing_df))
housing_df.columns = housing_df_columns
housing_df.index = housing_df_index

# Use LSTAT and PTRATIO as features
X = pd.DataFrame(np.c_[housing_df["LSTAT"], housing_df["PTRATIO"]], columns=[
                 "LSTAT", "PTRATIO"])
Y = housing_df["MEDV"]

# Split into 70% training set, 30% testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# Q1 part 1
print("Q1 with Gradient Descent" + "\n" +
      "------------------------")
q1_1 = LinearReg_GD()
q1_1.fit(X_train, Y_train)
q1_1_train_predictions = q1_1.predict(X_train)

# Model evaluation for training set
print("Root Mean Squared Error (RMSE) on training set:\n", np.sqrt(
    q1_1.mean_squared_error(Y_train, q1_1_train_predictions)), "\n", sep="")

# Model evaluation for testing set
q1_1_test_predictions = q1_1.predict(X_test)

print("Root Mean Squared Error (RMSE) on testing set:\n", np.sqrt(
    q1_1.mean_squared_error(Y_test, q1_1_test_predictions)), "\n", sep="")


# Q1 part 2
print("Q1 with Stochastic Gradient Descent" + "\n" +
      "-----------------------------------")
q1_2 = LinearReg_SGD()
q1_2.fit(X_train, Y_train)
q1_2_train_predictions = q1_2.predict(X_train)

# Model evaluation for training set
print("Root Mean Squared Error (RMSE) on training set:\n", np.sqrt(
    q1_2.mean_squared_error(Y_train, q1_2_train_predictions)), "\n", sep="")

# Model evaluation for testing set
q1_2_test_predictions = q1_2.predict(X_test)

print("Root Mean Squared Error (RMSE) on testing set:\n", np.sqrt(
    q1_2.mean_squared_error(Y_test, q1_2_test_predictions)), "\n", sep="")


# Q1 part 3
print("Q1 with Stochastic Gradient Descent with Momentum" + "\n" +
      "-------------------------------------------------")
q1_3 = LinearReg_SGD_Momentum()
q1_3.fit(X_train, Y_train)
q1_3_train_predictions = q1_3.predict(X_train)

# Model evaluation for training set
print("Root Mean Squared Error (RMSE) on training set:\n", np.sqrt(
    q1_3.mean_squared_error(Y_train, q1_3_train_predictions)), "\n", sep="")

# Model evaluation for testing set
q1_3_test_predictions = q1_3.predict(X_test)

print("Root Mean Squared Error (RMSE) on testing set:\n", np.sqrt(
    q1_3.mean_squared_error(Y_test, q1_3_test_predictions)), "\n", sep="")


# Q1 part 4
print("Q1 with Stochastic Gradient Descent with Nesterov Momentum" + "\n" +
      "-------------------------------------------------")
q1_4 = LinearReg_SGD_Nesterov()
q1_4.fit(X_train, Y_train)
q1_4_train_predictions = q1_3.predict(X_train)

# Model evaluation for training set
print("Root Mean Squared Error (RMSE) on training set:\n", np.sqrt(
    q1_4.mean_squared_error(Y_train, q1_4_train_predictions)), "\n", sep="")

# Model evaluation for testing set
q1_4_test_predictions = q1_4.predict(X_test)

print("Root Mean Squared Error (RMSE) on testing set:\n", np.sqrt(
    q1_4.mean_squared_error(Y_test, q1_4_test_predictions)), "\n", sep="")


# Q1 part 5
print("Q1 with AdaGrad" + "\n" +
      "---------------")
q1_5 = LinearReg_AdaGrad()
q1_5.fit(X_train, Y_train)
q1_5_train_predictions = q1_5.predict(X_train)

# Model evaluation for training set
print("Root Mean Squared Error (RMSE) on training set:\n", np.sqrt(
    q1_5.mean_squared_error(Y_train, q1_5_train_predictions)), "\n", sep="")

# Model evaluation for testing set
q1_5_test_predictions = q1_5.predict(X_test)

print("Root Mean Squared Error (RMSE) on testing set:\n", np.sqrt(
    q1_5.mean_squared_error(Y_test, q1_5_test_predictions)), "\n", sep="")

print("---------------------------------------------------" + "\n")

# Q2
titanic_df = pd.read_csv("titanic3.csv")
titanic_df.head()


# Impute the null values for age column
def impute_age(columns):
    age = columns[0]
    pclass = columns[1]

    if pd.isnull(age):
        return round(titanic_df[titanic_df["pclass"] == pclass]["age"].mean())
    return age


titanic_df["age"] = titanic_df[["age", "pclass"]].apply(impute_age, axis=1)
titanic_df.drop("cabin", axis=1, inplace=True)

# Remove unnecessary columns and convert certain categorical variables to one-hot representations
sex_col = pd.get_dummies(titanic_df["sex"], drop_first=True)
embarked_col = pd.get_dummies(titanic_df["embarked"], drop_first=True)

titanic_df.drop(["name", "ticket", "sex", "embarked",
                 "home.dest", "body", "boat"], axis=1, inplace=True)
titanic_df = pd.concat([titanic_df, sex_col, embarked_col], axis=1)
titanic_df.dropna(axis=0, inplace=True)

# Use the Standard Scaler on the data
std_scaler = StandardScaler()
X = titanic_df.drop("survived", axis=1)
X_columns = X.columns
X_index = X.index
X_scaled = pd.DataFrame(std_scaler.fit_transform(X))
X_scaled.columns = X_columns
X_scaled.index = X_index
titanic_df = pd.concat([X_scaled, titanic_df["survived"]], axis=1)
titanic_df.head()

# Split into 80% training set, 20% testing set
X = titanic_df.drop("survived", axis=1)
Y = titanic_df["survived"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Q2 part 1
print("Q2 with Gradient Descent" + "\n" +
      "------------------------")
q2_1 = LogisticReg_GD()
q2_1.fit(X_train, Y_train)

q2_1_train_predictions = q2_1.predict(X_train)

# Model evaluation for training set
print("Accuracy on training set:\n", q2_1.accuracy_score(
    Y_train, q2_1_train_predictions), "\n", sep="")
print(q2_1.classification_report(Y_train, q2_1_train_predictions), end="\n\n")

# Model evaluation for testing set
q2_1_test_predictions = q2_1.predict(X_test)

print("Accuracy on testing set:\n", q2_1.accuracy_score(
    Y_test, q2_1_test_predictions), "\n", sep="")
print(q2_1.classification_report(Y_test, q2_1_test_predictions), end="\n\n")


# Q2 part 2
print("Q2 with Stochastic Gradient Descent" + "\n" +
      "------------------------")
q2_2 = LogisticReg_SGD()
q2_2.fit(X_train, Y_train)

q2_2_train_predictions = q2_2.predict(X_train)

# Model evaluation for training set
print("Accuracy on training set:\n", q2_2.accuracy_score(
    Y_train, q2_2_train_predictions), "\n", sep="")
print(q2_2.classification_report(Y_train, q2_2_train_predictions), end="\n\n")

# Model evaluation for testing set
q2_2_test_predictions = q2_2.predict(X_test)

print("Accuracy on testing set:\n",
      q2_2.accuracy_score(Y_test, q2_2_test_predictions), "\n", sep="")
print(q2_2.classification_report(Y_test, q2_2_test_predictions), end="\n\n")


# Q2 part 3
print("Q2 with Stochastic Gradient Descent with Momentum" + "\n" +
      "------------------------")
q2_3 = LogisticReg_SGD_Momentum()
q2_3.fit(X_train, Y_train)

q2_3_train_predictions = q2_3.predict(X_train)

# Model evaluation for training set
print("Accuracy on training set:\n", q2_3.accuracy_score(
    Y_train, q2_3_train_predictions), "\n", sep="")
print(q2_3.classification_report(Y_train, q2_3_train_predictions), end="\n\n")

# Model evaluation for testing set
q2_3_test_predictions = q2_3.predict(X_test)

print("Accuracy on testing set:\n", q2_3.accuracy_score(
    Y_test, q2_3_test_predictions), "\n", sep="")
print(q2_3.classification_report(Y_test, q2_3_test_predictions), end="\n\n")


# Q2 part 4
print("Q2 with Stochastic Gradient Descent with Nesterov Momentum" + "\n" +
      "------------------------")
q2_4 = LogisticReg_SGD_Nesterov()
q2_4.fit(X_train, Y_train)

q2_4_train_predictions = q2_4.predict(X_train)

# Model evaluation for training set
print("Accuracy on training set:\n", q2_4.accuracy_score(
    Y_train, q2_4_train_predictions), "\n", sep="")
print(q2_4.classification_report(Y_train, q2_4_train_predictions), end="\n\n")

# Model evaluation for testing set
q2_4_test_predictions = q2_4.predict(X_test)

print("Accuracy on testing set:\n", q2_4.accuracy_score(
    Y_test, q2_4_test_predictions), "\n", sep="")
print(q2_4.classification_report(Y_test, q2_4_test_predictions), end="\n\n")


# Q2 part 5
print("Q2 with AdaGrad" + "\n" +
      "------------------------")
q2_5 = LogisticReg_AdaGrad()
q2_5.fit(X_train, Y_train)

q2_5_train_predictions = q2_5.predict(X_train)

# Model evaluation for training set
print("Accuracy on training set:\n", q2_5.accuracy_score(
    Y_train, q2_5_train_predictions), "\n", sep="")
print(q2_5.classification_report(Y_train, q2_5_train_predictions), end="\n\n")

# Model evaluation for testing set
q2_5_test_predictions = q2_5.predict(X_test)

print("Accuracy on testing set:\n", q2_5.accuracy_score(
    Y_test, q2_5_test_predictions), "\n", sep="")
print(q2_5.classification_report(Y_test, q2_5_test_predictions), end="\n\n")
