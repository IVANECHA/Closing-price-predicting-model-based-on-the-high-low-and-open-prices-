from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import math

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

dataset = pd.read_csv('simp_google_finance_data.csv')

x = dataset[['High', 'Low', 'Open']].values
y = dataset['Close'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
#print(regressor.coef_) 

print(regressor.intercept_)
predicted = regressor.predict(x_test)
#print(predicted)
dframe = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted': predicted.flatten()})
print(dframe.tail(25))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))
print('Mean Square Error:', metrics.mean_squared_error(y_test, predicted))
print('Root Mean Sqauer Error:', math.sqrt(metrics.mean_squared_error(y_test,predicted)))

graph = dframe.head(20)
graph.plot(kind='bar')
plt.show()