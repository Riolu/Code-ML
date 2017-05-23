import csv
import numpy as np
from matplotlib import pyplot as plt

price = []
living_raw = []
csvfile = file('train.csv', 'rb')
reader = csv.reader(csvfile)
for line in reader:
    if line[2]=="price":
        continue;
    price.append(int(line[2]))
    living_raw.append(int(line[5]))
csvfile.close()
one = np.ones(17289)
living = np.c_[living_raw,one]

test_price = []
test_living = []
csvfile = file('test.csv', 'rb')
reader = csv.reader(csvfile)
for line in reader:
    if line[2]=="price":
        continue;
    test_price.append(int(line[2]))
    test_living.append(int(line[5]))
csvfile.close()

a = np.random.normal()
b = np.random.normal()
num_fe = 1
H = np.zeros((num_fe, num_fe))
for x in living:
    H = H + np.array(np.matrix(x).T*np.matrix(x))
inv_H = np.array(np.matrix(H).I)


def h(x):
    return a*x+b

def error():
    E = np.zeros((1,2))
    for x,y in zip(living, price):
       # print(x)
        E = E + (h(x)-y)*x
    return E

def newton():
    global a,b
    e = error()
    a = a - np.array(np.matrix(inv_H) * np.matrix(e[0]).T)[0]
    b = b - np.array(np.matrix(inv_H) * np.matrix(e[0]).T)[1]
 #       print(a, b)

def train():
    newton()

def disp():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)  
    ax1.set_title('Scatter Plot')
    plt.xlabel('sqft_living')  
    plt.ylabel('price') 
    plt.scatter(living_raw,price)
    plt.xlim(0,14000)
    plt.ylim(0,8e+06)

    X = np.linspace(0, 14000, 100, endpoint=True)
    Y = a*X+b
    plt.plot(X,Y)
    plt.show()

def test():
    RMSE = 0;
    for i in range(len(test_price)):
        tmp = a*test_living[i]+b
        RMSE = RMSE + (test_price[i]-tmp)**2
    RMSE = (RMSE/len(test_price))**.5
    return RMSE

def main():
    train()
    print(a,b)
    print test()
    disp()

main()
