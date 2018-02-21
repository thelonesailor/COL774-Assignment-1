import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

# read the data
csvfile=open("weightedX.csv", 'r')
x = list(csv.reader(csvfile))
csvfile=open("weightedY.csv", 'r')
y = list(csv.reader(csvfile))


m=len(x)
n=1
x3=[]
y2=[]
for i in range(m):
    x3.append(float(x[i][0]))
    y2.append(float(y[i][0]))


# normalise the data
meanx=sum(x3)/m
v=0 # variance
for i in range(m):
    t=x3[i]-meanx
    v+=t*t
v/=m
v=math.sqrt(v)
for i in range(m):
    x3[i]=(x3[i]-meanx)/v

x2=[]
for i in range(m):
    x2.append(np.array([1,x3[i]]))


X=np.array(x2)
Y=np.array(y2)

xvalues=np.linspace(min(x3),max(x3),100)

plt.ion()
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

# plots Training data &
# straight line from linear regression
def pl(th):
    ax1.clear()
    ax1.scatter(x3, y2, label= "Training Data", color= "r",
                marker= ".", s=10)

    the=list(th)
    yvalues=the[1]*xvalues+the[0]

    ax1.plot(xvalues, yvalues, label="Hypothesis function learned",color ='b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Q2 (a)')
    plt.show()
    plt.pause(0.001)


# All weights same
# theta= inv(X'*X)*X'*Y
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T ,X)) , np.transpose(X)) , Y)
print("theta=",theta)

plt.ioff()
pl(theta)

# Part (b)
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

# change value of tau for part (c)
tau=0.8
tau2=tau*tau

# plots the hypothesis function learned
def plot_2():
    ax1.clear()
    ax1.scatter(x3, y2, label= "Training Data", color= "r",
                marker= ".", s=10)

# calculate the yaxis values for corresponding xaxis values
    yvalues=[]
    for i in range(len(xvalues)):

        weights=[]
        for j in range(m):
            c=xvalues[i]-X[j][1]
            power=-(c*c)/(2*tau2)
            weights.append(math.exp(power))

# convert np array to diagonal matrix
# W is m*m matrix
        W=np.diag(np.array(weights))

        # theta=inv(X'*W*X)*X'*W*Y
        the = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T ,W),X)) , X.T), W) , Y)
        yvalues.append(the[1]*xvalues[i]+the[0])

    ax1.plot(xvalues, yvalues, label="Hypothesis function learned",color ='b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Q2  tau={}'.format(tau))
    plt.show()
    plt.pause(0.001)

plt.ioff()
plot_2()
