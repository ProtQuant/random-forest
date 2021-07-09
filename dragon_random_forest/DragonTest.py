#example code to try and predict observed 'heartbeat' based on Dragon chemical properties
#using sci-kit-learn Random Forest
#note - it doesn't predict very well at all!

import csv
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

num_trees= 2000

input_file = "dragon_161.txt"

with open(input_file, "r") as source:
    rdr = csv.reader(source, delimiter="\t")
    #just get the header line from the input file so can index columns by the header value
    header_line = next(rdr)
    head = {k: v for v, k in enumerate(header_line)}
    # print(head)

    #temporary store of numerical data so it can be turned into a numpy array
    numerical_data = []

    for row in rdr:

        # print(row)
        # print(row[0])
        #just set the non-numerical values to 0
        row[0] = 0
        row[head["Smile"]] = 0

        newrow=np.array(row)

        # also convert NA to 0
        newrow[newrow=="NA"] = 0

        newrow = newrow.astype(np.float)
        numerical_data.append(newrow)

numerical_data = np.array(numerical_data)

Rf = RandomForestRegressor(n_estimators=num_trees, max_depth=None)

#this is the data we are going to try and train for
target_column = head["heartbeat"]

#split the data rows up randomly - we will train on some, and test on the others
train,test = train_test_split(numerical_data)

#sort the test data by heartbeat, only to make the charts a bit nicer.
test = test[test[:,target_column].argsort()]


#take out the value we are training for
train_target = train[:, target_column]
train_data = np.delete(train, target_column, 1)

test_target = test[:, target_column]
test_data = np.delete(test, target_column, 1)

#fit the Random Forest, and run the prediction
Rf.fit(train_data,train_target)
test_predictions = Rf.predict(test_data)

print(test_predictions, test_target)

#get an R^2 score
print(r2_score(test_predictions,test_target ))

from matplotlib import pyplot as plt
x = range(len(test_target))
plt.scatter(x, test_target )
plt.scatter(x, test_predictions )
plt.show()

