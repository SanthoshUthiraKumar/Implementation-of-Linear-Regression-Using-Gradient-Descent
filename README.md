# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Importing the Standard Libraries.

2. Uploading the text file to a python compiler.

3. Obtaining the computeCost,h(x) of the data from the text file.

4. Plotting the Graphs and finding the profit for the Population 35,000 and 70,000.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Santhosh U
RegisterNumber:  212222240092
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
 m=len(y)
 J_history=[]
 for i in range(num_iters):
   predictions=x.dot(theta)
   error=np.dot(x.transpose(),(predictions-y))
   descent=alpha*1/m*error
   theta-=descent
   J_history.append(computeCost(x,y,theta))
 return theta,J_history
 
theta,J_history=gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict (x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population =70,000,we predict a profit of $"+str(round(predict2,0)))

```

## Output:
### 1.Profit Prediction graph
![Output1](https://user-images.githubusercontent.com/119477975/229809489-ea2e91c3-fda6-419c-a8d8-053985c61610.png)

### 2.Compute Cost Value
![Output2](https://user-images.githubusercontent.com/119477975/229809520-a47c1215-a694-4a09-845c-42a10738eeeb.png)

### 3.h(x) Value
![Output3](https://user-images.githubusercontent.com/119477975/229809564-2065d1e1-4b96-47d2-8fa7-e8f3b6b85dbe.png)

### 4.Cost function using Gradient Descent Graph
![Output4](https://user-images.githubusercontent.com/119477975/229809597-c5e87da9-0280-4d8d-8701-2d75d308fa5d.png)

### 5.Profit Prediction Graph
![Output5](https://user-images.githubusercontent.com/119477975/229809697-bd90493f-6ad3-4937-beee-bcaaee4f7635.png)

### 6.Profit for the Population 35,000
![Output6](https://user-images.githubusercontent.com/119477975/229809731-0e94f914-fb5a-4e5e-bd02-4a5a03956ca6.png)

### 7.Profit for the Population 70,000
![Output7](https://user-images.githubusercontent.com/119477975/229809754-78ac9a69-9eef-4880-b050-23efd393513a.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
