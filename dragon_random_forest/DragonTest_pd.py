#example code to try and predict observed 'heartbeat' based on Dragon chemical properties
#using sci-kit-learn Random Forest
#note - it doesn't predict very well at all!

import csv
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

num_trees= 200

input_file = "dragon_161.txt"

df = pd.read_csv(input_file,sep='\t')
#just replace tne 'NA' with 0
df = df.fillna(0)

Rf = RandomForestRegressor(n_estimators=num_trees, max_depth=None)
#removing the text columns
df = df._get_numeric_data()

#this is the data we are going to try and train for
label = 'heartbeat'

#split the data rows up randomly - we will train on some, and test on the others
train, test = train_test_split(df)
X_test = train[label]
X_train = train.drop(label, 1)

#just sorting to make the charts easier to understand
test = test.sort_values(by=[label] )
Y_test = test[label]
Y_train = test.drop(label, 1)


#fit the Random Forest, and run the prediction
Rf.fit(X_train,X_test)
test_predictions = Rf.predict(Y_train)

print(test_predictions, Y_test)

#get an R^2 score
print(r2_score(test_predictions,Y_test ))

from matplotlib import pyplot as plt
x = range(len(Y_test))
plt.scatter(x, Y_test )
plt.scatter(x, test_predictions )
plt.show()

