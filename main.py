import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score


#import the data from Url
URL="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
#read the data
data=pd.read_csv(URL)
#print(data)

#Visualize the data and plotting it
data.plot(x='Hours',y='Scores',style='x',color='m')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
#Data Preparing
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)

#Modeling
Linear=LinearRegression()
Linear.fit(xtrain,ytrain)

# Plotting the regression line
line = Linear.coef_*X+Linear.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

#predicting
y_pre=Linear.predict(xtest)

# if the hours = 9.25
own_pred = Linear.predict([[9.25]])
print("No of Hours = {}".format(9.25))
print("Predicted Score = {}".format(own_pred[0]))
#print(y_pre)

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': ytest, 'Predicted': y_pre})
print(df)

#Evaluating the model
print("Mean Error = {}".format(mean_absolute_error(ytest,y_pre)))
print("Accuracy = {:%}".format(r2_score(ytest,y_pre)))

