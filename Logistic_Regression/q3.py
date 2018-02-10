import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from scipy.special import expit


csvfile=open("logisticX.csv", 'r')
x = list(csv.reader(csvfile))
csvfile=open("logisticY.csv", 'r')
y = list(csv.reader(csvfile))

# print(len(x))
# print(len(y))

m=len(x)
n=2
x3=[]
y2=[]
for i in range(m):
    x3.append([float(x[i][0]),float(x[i][1])])
    y2.append(float(y[i][0]))


# normalise
meanx=[0,0]
meanx[0]=sum(x3[0])/m
meanx[1]=sum(x3[1])/m

v=[0,0]
for i in range(m):
    temp=x3[i][0]-meanx[0]
    v[0]+=temp*temp
    temp=x3[i][1]-meanx[1]
    v[1]+=temp*temp
v[0]/=m
v[0]=math.sqrt(v[0])
v[1]/=m
v[1]=math.sqrt(v[1])
for i in range(m):
    x3[i][0]=(x3[i][0]-meanx[0])/v[0]
    x3[i][1]=(x3[i][1]-meanx[1])/v[1]


x2=[]
x4=[]
x5=[]
x6=[]
x7=[]
for i in range(m):
    x2.append(np.array([1,x3[i][0],x3[i][1]]))

    if(y2[i]==1):
        x4.append(float(x3[i][0]))
        x5.append(float(x3[i][1]))
    else:
        x6.append(float(x3[i][0]))
        x7.append(float(x3[i][1]))

X=np.array(x2)
Y=np.array(y2)
# print(X.shape)
# print(Y.shape)




epsilon=0.000001
t=0
theta=np.array([[0],[0],[0]])
theta=np.array([0,0,0])


while(True):

    X3=np.matmul(X,theta)
    X2=expit(np.matmul(X,theta))


    gradient=np.dot(X.T,(Y-X2))
    # print(gradient.shape)

    H=np.zeros((n+1,n+1))
    for j in range(m):
        x2=X3[j]
        temp=np.outer(X[j].T,X[j])
        # print(temp)
        # print()
        temp=expit(x2)*(1-expit(x2))
        # print(x2)
        H=H-((expit(x2)*(1-expit(x2)))*(np.outer(X[j].T,X[j])))
        # print(H)

    t+=1

    thetaold=theta
    # print(np.linalg.inv(H))
    # print(gradient)
    theta=theta-np.matmul(np.linalg.inv(H),gradient)

    temp=np.linalg.norm(theta-thetaold)
    # print(temp)
    if(temp<epsilon):
        break

print(t)
print(theta)


plt.ioff()
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

def pl():
    ax1.clear()
    ax1.scatter(x4, x5, label= "y=1 Training Data", color= "red",
                marker= ".", s=10)
    ax1.scatter(x6, x7, label= "y=0 Training Data", color= "blue",
                marker= ".", s=10)

    xvalues=xvalues=np.arange(min(min(x4),min(x6))-0.2,max(max(x4),max(x6))+0.2,0.1)

    the=theta
    yvalues=-1*(the[1]*xvalues+the[0])/the[2]

    ax1.plot(xvalues, yvalues, label="Decision boundary learned",color ="#66cc00")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Q3 (b)')
    plt.show()
    plt.pause(0.001)

pl()
