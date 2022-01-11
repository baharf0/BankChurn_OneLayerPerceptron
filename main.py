
import numpy as np
import pandas as pd

learning_rate = 0.1
nodes = 8
epoch = 1000

data1 = pd.read_csv('Churn_Data.csv', sep=',')
desired = data1['Exited']
no_need_columns = [0, 1, 2, 13]
data2 = data1.drop(data1.columns[no_need_columns], axis=1)
data3 = data2.replace(['Male', 'Female'], [0, 1])
country = data3['Geography'].unique()
country_number = []
for i in range(0, len(country)):
    country_number.append(i)
data4 = data3.replace(country, country_number)
data = data4.apply(lambda x: (x - x.min())/(x.max()-x.min()), axis=0)
train = data.head(int(len(data)*0.8))  # 7200x10
train_desired = desired.head(int(len(data)*0.8))  # 7200x1
test = data.drop(train.index)  # 1800x10
test_desired = desired.drop(train.index)  # 1800x1


def sigmoid(net):
    return 1/(1+np.exp(-net))

# feedforward


bias1 = 1 * np.random.uniform(0, 1)  # 1x1
bias2 = 1 * np.random.uniform(0, 1)  # 1x1
output = [[0]] * len(train)  # 7200x1
w1 = np.random.uniform(-1, 1, (len(train.columns), nodes))  # 10x8
h = sigmoid(np.dot(train, w1) + bias1)  # 7200x8
w2 = np.random.uniform(-1, 1, (nodes, 1))  # 8x1
f_net = sigmoid(np.dot(h, w2) + bias2)  # 7200x1
for j in range(0, len(f_net)):
    output[j] = 1 if f_net[j] >= 0.5 else 0  # 7200x1
error = train_desired - output  # 7200x1
accurate_count = len(train) - np.count_nonzero(error)  # 7200x1
accuracy = accurate_count / len(train) * 100  # 1x1
print("accuracy before update: ", accuracy)

# backpropagation
w1_new = w1
w2_new = w2
bias1_new = bias1
bias2_new = bias2
h_new = h
f_net_new = f_net
output_new = [[0]] * len(train)  # 7200x1
for e in range(0, epoch):
    delta1 = [[]]  # 7200x8
    delta2 = [[]]  # 7200x1
    delta2 = f_net_new * (1-f_net_new) * (np.array(train_desired).reshape(len(train), 1) - f_net_new)
    delta1 = h_new * (1 - h_new) * np.dot(delta2, np.transpose(w2_new))
    w2_new = w2_new + learning_rate * np.dot(np.transpose(h_new), delta2)  # 8x1
    w1_new = w1_new + learning_rate * np.dot(np.transpose(train), delta1)  # 10x8
    bias2_new = learning_rate * delta1 * bias2_new
    bias1_new = learning_rate * delta2 * bias1_new
    h_new = sigmoid(np.dot(train, w1_new)) + bias1_new  # 7200x8
    f_net_new = sigmoid(np.dot(h_new, w2_new))  # 7200x1
    for k in range(0, len(f_net)):
        output_new[k] = 1 if f_net_new[k] >= 0.5 else 0
    error_new = train_desired - output_new
    accurate_count_new = len(train) - np.count_nonzero(error_new)
    accuracy_new = accurate_count_new / len(train) * 100
    print("number of correct: ", accurate_count_new)
    print("accuracy after", e+1, "update: ", accuracy_new)















