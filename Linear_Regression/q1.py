import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

# read the data
csvfile=open("linearX.csv", 'r')
x = list(csv.reader(csvfile))
csvfile=open("linearY.csv", 'r')
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

xvalues=np.arange(min(x3)-1,max(x3)+1,0.1)

plt.ion()
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

# plots hypothesis funtion and training data
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

# return the value of J(theta)
def findcost(th):
    ans=0
    for i in range(m):
        temp=(Y[i]-np.sum(np.dot(th,X[i])))
        ans+=(temp*temp)
    ans=float(ans/2)
    return ans

def find_convergence(th1,th2):
    return abs(findcost(th1)-findcost(th2))

# Part (a)&(b)
epsilon=0.0000001
# epsilon=5
eta=0.001
theta=np.array([0,0])
t=0

pl(theta)

while(True):

    hthetas=[]
    for i in range(m):
        hthetas.append(np.sum(np.dot(theta,X[i])))

# calculate -ve of gradient
    gradient=np.zeros((n+1))
    for i in range(m):
        gradient+=(Y[i]-hthetas[i])*X[i]

# update theta
    thetaold=theta
    theta=theta+eta*gradient

    t+=1

    print("theta={}\n".format(theta))

    pl(theta)

# check if converged
    convergence=find_convergence(thetaold,theta)
    print("convergence={}".format(convergence))
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
ax = fig.add_subplot(111, projection='3d')

# plot J(theta)
xvalues = np.linspace(0, 2, 70)
yvalues = np.linspace(-1, 1, 70)
xvalues, yvalues = np.meshgrid(xvalues, yvalues)
Zs = np.array([findcost(np.array([x,y])) for x,y in zip(np.ravel(xvalues), np.ravel(yvalues))])
Z = Zs.reshape(xvalues.shape)

surf = ax.plot_surface(xvalues, yvalues, Z, cmap=cm.coolwarm,linewidth=0,alpha=0.7 ,antialiased=False,label='J(theta)')
plt.show()
plt.pause(0.001)

# plots point in 3d mesh
def plot_2(th):

    ax.scatter(th[0],th[1],findcost(th),c="black",depthshade=False,s=6)

    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Q1 (c)')
    plt.show()
    plt.pause(0.2)
    # plt.pause(0.001)


theta=np.array([0,0])
t=0
levels=[] # stores the values of J(theta) at each iteration
levels.append(findcost(theta))
thetas=[]
thetas.append(theta)
while(True):
    hthetas=[]
    for i in range(m):
        hthetas.append(np.sum(np.dot(theta,X[i])))

    gradient=np.zeros((n+1))
    for i in range(m):
        gradient+=(Y[i]-hthetas[i])*X[i]
    thetaold=theta
    theta=theta+eta*gradient

    levels.append(findcost(theta))
    thetas.append(theta)

    t+=1

    print("theta={}\n".format(theta))

    plot_2(theta)

    convergence=find_convergence(thetaold,theta)
    print("convergence=",convergence)
    if(convergence<epsilon):
        break

print("Number of iterations=",t)
plt.ioff()
plot_2(theta)



# part (d)&(e)
# plots contours at each iteration

plt.ion()
plt.figure()

plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('eta={}'.format(eta))
for i in range(len(levels)):
    print(i)
    plt.contour(xvalues, yvalues, Z,levels[i])
    plt.scatter(thetas[i][0],thetas[i][1])

    plt.show()
    plt.pause(0.2)

plt.ioff()
plt.show()
