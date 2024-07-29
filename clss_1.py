import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


ver_d = {"hour_study":[2,3,4,5,6,7,8,9,10],"exam_scor":[50,60,70,75,80,85,90,92,95]}

ver = pd.DataFrame(ver_d)
print (ver.head(3))


x = ver[["hour_study"]]
y = ver[["exam_scor"]]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42 )

model = LinearRegression()

model.fit(x_train, y_train)


user = float(input("Enter the number of hours you study: "))

predicted = model.predict([[user]])

print (f"predicted exam scor: {predicted[0][0]:.2f}")