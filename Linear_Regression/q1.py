import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np



csvfile=open("linearX.csv", 'r')
x = list(csv.reader(csvfile))
csvfile=open("linearY.csv", 'r')
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

xvalues=np.arange(min(x3)-1,max(x3)+1,0.1)

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
    plt.title('Q1 (b)')
    plt.show()
    plt.pause(0.001)

# J(theta)
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

epsilon=0.0000001
# epsilon=5
eta=0.001
theta=np.array([0,0])
t=0

while(True):

    hthetas=[]
    for i in range(m):
        hthetas.append(np.sum(np.dot(theta,X[i])))

    temp=np.zeros((n+1))
    for i in range(m):
        t2=(Y[i]-hthetas[i])
        temp+=t2*X[i]
    theta2=theta
    theta=theta+eta*temp
    t+=1
    # print(t)

    print(theta2)
    print(theta)
    print()

    pl(theta)

    convergence=find_convergence(theta2,theta)
    print("convergence=",convergence)
    if(convergence<epsilon):
        break

print("Number of iterations=",t)
plt.ioff()
pl(theta)

# part (c)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


plt.ion()

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')


xvalues = np.linspace(0, 2, 130)
# xvalues = np.linspace(0.9, 1.1, 30)
yvalues = np.linspace(-1, 1, 130)
xvalues, yvalues = np.meshgrid(xvalues, yvalues)
# Z = findcost(np.array([xvalues,yvalues]))
Zs = np.array([findcost(np.array([x,y])) for x,y in zip(np.ravel(xvalues), np.ravel(yvalues))])
Z = Zs.reshape(xvalues.shape)

# print("xvalues=",xvalues)
# print("yvalues=",yvalues)
# print("Z=",Z)

surf = ax.plot_surface(xvalues, yvalues, Z, cmap=cm.coolwarm,linewidth=0,alpha=0.6 ,antialiased=False,label='J(theta)')
plt.show()
plt.pause(0.001)

def pl2(th):

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax1.plot(np.array([th[0]]), np.array([th[1]]), np.array([findcost(th)]), color= "b")
    ax.scatter(th[0],th[1],findcost(th),c="black",depthshade=False,s=5)

    plt.xlabel('theta0')
    plt.ylabel('theta1')
    # plt.legend()
    plt.title('Q1 (c)')
    plt.show()
    plt.pause(0.001)

# epsilon=1

theta=np.array([0,0])
t=0
thlist=[]
levels=[]
while(True):
    hthetas=[]
    for i in range(m):
        hthetas.append(np.sum(np.dot(theta,X[i])))

    temp=np.zeros((n+1))
    for i in range(m):
        t2=(Y[i]-hthetas[i])
        temp+=t2*X[i]
    theta2=theta
    theta=theta+eta*temp

    thlist.append(tuple(theta))
    levels.append(findcost(theta))

    t+=1
    # print(t)

    print(theta2)
    print(theta)
    print()

    pl2(theta)

    convergence=find_convergence(theta2,theta)
    print("convergence=",convergence)
    if(convergence<epsilon):
        break

print("Number of iterations=",t)
plt.ioff()
pl2(theta)



plt.ion()
plt.figure()

plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('eta=0.021')
for i in range(len(levels)):
    if(levels[i]<0):break
    CS = plt.contour(xvalues, yvalues, Z,[levels[i]])
    plt.show()
    plt.pause(0.02)

plt.ioff()
plt.show()
