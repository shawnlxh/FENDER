import numpy as np
import math
import pickle as pk
import sys
import argparse

parser = argparse.ArgumentParser(description='Choose dataset.')
parser.add_argument('--dataset', nargs='?', default='instacart',
                        help='Choose a dataset from {instacart, walmart}')

path = './Data/walmart/'
file1 = open(path+'train_mf.txt','r')

train_raw = []

for line in file1:
    line = line.split(' ')
    user = int(line[0])
    item = int(line[1])
    score = float(line[2])
    train_raw.append([user, item, score])

num_user = 0
num_item = 0
for line in train_raw:
    user = line[0]
    item = line[1]
    if user > num_user:
        num_user = user
    if item > num_item:
        num_item = item

print(num_user)
print(num_item)
if num_user > 100000 or num_item > 100000:
    print("data size is out of memory")
    sys.exit()

train_data = np.zeros((num_user+1,num_item+1))

user_p = np.random.randn(num_user+1,64)*0.5
item_q = np.random.randn(num_item+1,64)*0.5
lamda = 0.02
learning_rate = 0.01

for line in train_raw:
    user = line[0]
    item = line[1]
    score = line[2]
    train_data[user][item] = score
    


def avg(matrix):
    count = 0
    total = 0
    for i in range(943):
        for j in range(1682):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j]
    avg_total = total / count
    return avg_total
    
def user_bias(matrix, avg):
    bias = []
    for i in range(943):
        count = 0
        total = 0
        for j in range(1682):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j] - avg
        if count == 0:
            bias.append(count)
        else:
            bias.append(total/count)
    return bias
    
def item_bias(matrix, avg):
    bias = []
    for j in range(1682):
        count = 0
        total = 0
        for i in range(943):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j] - avg
        if count == 0:
            bias.append(count)
        else:
            bias.append(total/count)
    return bias

avg_train = avg(train_data)   #average of the training data
    
def train():
    bias_u1 = user_bias(train_data, avg_train)
    bias_i1 = item_bias(train_data, avg_train)
    for u in range(943):
        for i in range(1682):
            if train_data[u][i] != 0:
                rui = avg_train + bias_u1[u] + bias_i1[i] + np.dot(user_p[u], item_q[i].T)
                eui = train_data[u][i] - rui
                bias_u1[u] += learning_rate * (eui - lamda * bias_u1[u])
                bias_i1[i] += learning_rate * (eui - lamda * bias_i1[i])
                temp = user_p[u]
                user_p[u] += learning_rate * (eui * item_q[i] - lamda * user_p[u])
                item_q[i] += learning_rate * (eui * temp - lamda * item_q[i])
                

    
for i in range(50):
    print(i)
    train()
    

f_m = open(path+"fre_matrix.pkl", "wb")
pk.dump(user_p, f_m)
pk.dump(item_q, f_m)
f_m.close()
    