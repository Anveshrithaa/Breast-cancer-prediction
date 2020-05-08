# Breast cancer detection code that uses Machine Learning algorithm (K-Nearest Neighbors (K-NN)) to classify tumors as malignant(cancerous) or benign(non-cancerous) from various parameters

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('data.csv', header=None)
dataset= dataset.replace('?', np.nan)
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])

#Describe the dataset
df=dataset.describe()

#Histogram plotting for each variable of the dataset
dataset.hist(figsize = (10, 10))
plt.show()

#Scatter plot matrix
scatter_matrix(dataset, figsize = (18,18))
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Fitting SVM to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric= 'minkowski', p=2)
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy 
score=accuracy_score(y_test, y_pred)

#Example prediction
example= np.array([[1,1,2,1,5,1,3,1,1]])
example= example.reshape(len(example),-1)
prediction = classifier.predict(example)




