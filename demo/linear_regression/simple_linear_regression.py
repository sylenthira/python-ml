
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('data/Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into the training set and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Predicting the test result
y_pred= regression.predict(x_test)


# Visualising the training set result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color = 'blue')
plt.title('Training Set: Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regression.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


