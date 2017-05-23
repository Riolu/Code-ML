import numpy as np
import random

def normalize(data):
    return 0

def readFiles(file,alphabet): #read the data from files
    data = open(file)
    needed = []
    for line in data.readlines():
        tmp = line.strip('\n').split(",")
        tmp[1:]=[int(i) for i in tmp[1:]]
        if tmp[0] in alphabet:
            if tmp[0]==alphabet[0]:
                tmp[0]=0
            else:
                tmp[0]=1
            needed.append(tmp)
    return needed

def sigmoid(a):
    b = np.matrix([1.0/(1.0+np.math.exp(-i)) for i in np.array(a)[0]])
    return b

def get_predict(y,y_res): #judge if the predicted ans is the same with the fact
    loss =np.sum((y-y_res)**2)
    acc = np.abs(y-y_res)
    if np.round(y)[0]==y_res[0] and np.round(y)[1]==y_res[1]:
        is_acc=1
    else:
        is_acc=0
    #if np.sum(acc)>0.2:
    #    is_acc=0
    #else:
    #   is_acc=1
    return loss,is_acc

def neural_network(data,m,N):
    #The hidden layer size is N
    #input layer size is m
    # input X[m], output Y[2], hidden layer b[N]
    #parameters X->b: v[m][N], b->Y: w[N][2]
    # bias X->b: theta_v[2][N], b->Y: theta_w[1][2]

    a = 0.3 #learning rate
    v = np.matrix([[2*random.random()-1 for i in range(N)] for _ in range(m)]) #init
    w = np.matrix([[2*random.random()-1,2*random.random()-1] for i in range(N)]) #init
    theta_v = np.matrix([2*random.random()-1 for i in range(N)])
    theta_w = np.matrix([2*random.random()-1,2*random.random()-1])
    size_data = len(data)
    train_size = int(size_data*7/10)
    epoch = 150
    #print(size_data)

    for eph in range(epoch):
        a = a-0.002
        sum_loss = 0
        accuracy = train_size
        for line in data[0:train_size]:
            #Forwarding
            x = np.matrix(line[1:]) #input 1*m
            alpha = x*v  #hidden layer input 1*N
            b = sigmoid(alpha-theta_v) #hidden layer output 1*N
            beta = b*w #output layer input 1*2
            y = sigmoid(beta-theta_w).A[0]
            y_res = np.array([line[0],1-line[0]])
            loss,is_acc=get_predict(y,y_res)
            loss =np.sum((y-y_res)**2)
            sum_loss+=loss
            if(is_acc==0):
                accuracy-=1

            #back probagetion
            g = y*(np.array([1,1])-y)*(y_res-y) #g = -dE/dy*dy/dbeta
            theta_w.A[0] = theta_w.A[0] - a*g
            prev_w = np.array(w.A)
            for i in range(N):
                w.A[i][0]=w.A[i][0]+a*g[0]*b.A[0][i]
                w.A[i][1]=w.A[i][1]+a*g[1]*b.A[0][i]

            for i in range(N):
                eh = b.A[0][i]*(1-b.A[0][i])*np.sum(prev_w[i]*g) #eh=-dE/dbh*dbh/dalphah
                theta_v.A[0][i] = theta_v.A[0][i] - a*eh
                for j in range(m):
                    v.A[j][i] = v.A[j][i]+a*eh*x.A[0][j]

        #print (sum_loss,float(accuracy)/train_size)
        print float(accuracy)/train_size*100 ,'%'


    #predict on the remaining data (30% of the data)
    sum_loss=0
    accuracy = size_data-train_size
    for line in data[train_size:]:
        #Forwarding
        x = np.matrix(line[1:]) #input 1*m
        alpha = x*v  #hidden layer input 1*N
        b = sigmoid(alpha-theta_v) #hidden layer output 1*N
        beta = b*w #output layer input 1*2
        y = sigmoid(beta-theta_w).A[0]
        y_res = np.array([line[0],1-line[0]])
        loss,is_acc = get_predict(y,y_res)
        sum_loss+=loss
        if(is_acc==0):
            accuracy-=1
        #print(loss,acc)
    #print (sum_loss,float(accuracy)/(size_data-train_size))
    print float(accuracy)/(size_data-train_size)*100,'%'

def main():
    data = readFiles("data/letter-recognition.data",['O','D'])
    #data = readFiles("data/letter-recognition.data",['O','X'])
    neural_network(data,16,5)
    return 0

if __name__=="__main__":
    main()
