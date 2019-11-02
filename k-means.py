# Importing libraries
import pandas as pd
import numpy as np

data=pd.read_csv("file:///C:/Users/STUDENT/Desktop/iris-species/Iris.csv")
data.head()

A=data.isnull().sum()
A

## Dropping the species column
data1=data.drop("Species",axis=1)
data1.head()

from sklearn.model_selection import train_test_split

X= data1
Y=data["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

## Model Building
from sklearn.cluster import KMeans
import sklearn.metrics as sm

model = KMeans(n_clusters=3)
model.fit(X_train, Y_train)

# View the results
# Set the size of the plot
import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
 
# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X.PetalLength, X.PetalWidth, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
 
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X_train.PetalLengthCm, X_train.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')

## Predicting on test dataset
Y_pred=KMeans.predict()
