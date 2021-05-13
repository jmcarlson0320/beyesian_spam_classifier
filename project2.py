# ml_programming_2
# Joshua Carlson
# 5/13/21
import bayes
import statistics
import math
import sys
import numpy as np
from enum import IntEnum

MIN_FLOAT = sys.float_info.min

# load and shuffle data
data = np.loadtxt(open('data/spambase.data'), delimiter=',')
np.random.shuffle(data)

# split into training and test sets
num_datapoints = data.shape[0]
test = data[:num_datapoints // 2]
train = data[num_datapoints // 2:]

# split into x and y arrays
x_train = test[:,:-1]
y_train = test[:,-1]
x_test = train[:,:-1]
y_test = train[:,-1]

# compute the model using training data
#       Compute prior for each class
num_spam = np.count_nonzero(y_train)
prior_spam = num_spam / y_train.shape[0]
prior_notspam = 1 - prior_spam

# enums for indexing into the model array
class Class(IntEnum):
    SPAM = 0
    NOTSPAM = 1

class Stat(IntEnum):
    MEAN = 0
    STDEV = 1

# this holds all the stats calculated in training, model[feature][class] returns a
# tuple containing (mean, stdev)
model = np.zeros((57, 2), dtype = ('float', 2))
# This can't be the most efficient way to do this... but I want to get something working first
for i in range(57):
    # these lists hold the feature data, split into classes
    # mean and stdev are calculated on these
    spam = []
    notspam = []
    # fill in the spam/notspam data lists
    for row in train:
        if row[-1] == 1:
            spam.append(row[i])
        else:
            notspam.append(row[i])

    # use lists to calculate mean and stdev for spam, store in the model for lookup later
    model[i][Class.SPAM][Stat.MEAN] = statistics.mean(spam)
    model[i][Class.SPAM][Stat.STDEV] = statistics.stdev(spam)
    if model[i][Class.SPAM][Stat.STDEV] == 0:
        model[i][Class.SPAM][Stat.STDEV] = 0.0001

    # use lists to calculate mean and stdev for notspam
    model[i][Class.NOTSPAM][Stat.MEAN] = statistics.mean(notspam)
    model[i][Class.NOTSPAM][Stat.STDEV] = statistics.stdev(notspam)
    if model[i][Class.NOTSPAM][Stat.STDEV] == 0:
        model[i][Class.NOTSPAM][Stat.STDEV] = 0.0001


# run the classifier on the test data
# use the log of the products to avoid underflow
num_correct = 0 # for calculating accuracy
for data_point, target_class in zip(x_test, y_test):

    prediction = [0, 0] # [spam, notspam]

    # this repeated code should be a function...
    # spam
    prediction[Class.SPAM] = math.log(prior_spam)
    for i in range(57):
        x = data_point[i]
        u = model[i][Class.SPAM][Stat.MEAN]
        s = model[i][Class.SPAM][Stat.STDEV]
        p_x_c = 1 / (math.sqrt(2 * math.pi) * s) * math.exp(-((x - u) ** 2) / (2 * s ** 2))
        if p_x_c == 0: # sometimes this is zero, not good when taking log...
            p_x_c = MIN_FLOAT
        p_x_c = math.log(p_x_c)
        prediction[Class.SPAM] += p_x_c

    # notspam
    prediction[Class.NOTSPAM] = math.log(prior_notspam)
    for i in range(57):
        x = data_point[i]
        u = model[i][Class.NOTSPAM][Stat.MEAN]
        s = model[i][Class.NOTSPAM][Stat.STDEV]
        p_x_c = 1 / (math.sqrt(2 * math.pi) * s) * math.exp(-((x - u) ** 2) / (2 * s **2))
        if p_x_c == 0:
            p_x_c = MIN_FLOAT
        p_x_c = math.log(p_x_c)
        prediction[Class.NOTSPAM] += p_x_c
        
    predicted_class = np.argmax(prediction)
    # needed because I did the enum backwards.. .will fix
    if predicted_class == Class.SPAM:
        predicted_class = 1
    else:
        predicted_class = 0

    if predicted_class == target_class:
        num_correct += 1

print('accuracy on the test set:')
print(num_correct / x_test.shape[0])
