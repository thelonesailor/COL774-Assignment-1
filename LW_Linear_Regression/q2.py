import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np



csvfile=open("weightedX.csv", 'r')
x = list(csv.reader(csvfile))
csvfile=open("weightedY.csv", 'r')
y = list(csv.reader(csvfile))

# print(len(x))
# print(len(y))

m=len(x)
n=1
x3=[]
y2=[]
for i in range(m):
    x3.append(float(x[i][0]))
    y2.append(float(y[i][0]))

# normalise
meanx=sum(x3)/m
v=0
for i in range(m):
    temp=x3[i]-meanx
    v+=temp*temp
v/=m
v=math.sqrt(v)
for i in range(m):
    x3[i]=(x3[i]-meanx)/v

x2=[]
for i in range(m):
    x2.append(np.array([1,x3[i]]))

X=np.array(x2)
Y=np.array(y2)
# print(X)
# print(Y)

xvalues=np.arange(min(x3)-0.2,max(x3)+0.2,0.1)

plt.ion()
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

def pl(th):
    ax1.clear()
    ax1.scatter(x3, y2, label= "Training Data", color= "r",
                marker= ".", s=10)

    the=list(th)
    yvalues=the[1]*xvalues+the[0]

    ax1.plot(xvalues, yvalues, label="Hypothesis function learned",color ='b')

    plt.xlabel('Acidity')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Q2 (a)')
    plt.show()
    plt.pause(0.001)

# J(theta) for linear regression
def findcost(th):
    ans=0
    for i in range(m):
        # print(th)
        temp=(Y[i]-np.sum(np.dot(th,X[i])))
        ans+=(temp*temp)
        # print(type(ans))
    ans=float(ans/2)
    return ans

def find_convergence(th1,th2):
    return abs(findcost(th1)-findcost(th2))

theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T ,X)) , np.transpose(X)) , Y)

plt.ioff()
pl(theta)

# (b)
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

tau=10
tau2=tau*tau

def pl2():
    ax1.clear()
    ax1.scatter(x3, y2, label= "Training Data", color= "r",
                marker= ".", s=10)


    yvalues=[]
    for i in range(len(xvalues)):

        weights=[]
        for j in range(m):
            c=xvalues[i]-X[j][1]
            temp=(c*c)/(2*tau2)
            temp=math.exp(-temp)
            weights.append(temp)

        weights=np.array(weights)
        W=np.diag(weights)

        the = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T ,W),X)) , np.transpose(X)), W) , Y)
        temp=the[1]*xvalues[i]+the[0]
        yvalues.append(temp)

    ax1.plot(xvalues, yvalues, label="Hypothesis function learned",color ='b')

    plt.xlabel('Acidity')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Q2 (b)  tau=10')
    plt.show()
    plt.pause(0.001)

pl2()
