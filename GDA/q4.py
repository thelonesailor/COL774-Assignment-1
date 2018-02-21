import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from scipy.special import expit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# read the data
csvfile=open("q4x.dat", 'r')
x = list(csv.reader(csvfile,delimiter=' '))
csvfile=open("q4y.dat", 'r')
y = list(csv.reader(csvfile))

# Alaska is 0 and Canada is 1
m=len(x)
n=2
x3=[]
y2=[]
for i in range(m):
    x3.append([float(x[i][0]),float(x[i][1])])
    if(y[i][0]=="Alaska"):
        y2.append(0)
    else:
        y2.append(1)


# normalise the data
meanx=[0,0]
# meanx[0]=sum(x3[0])/m
# meanx[1]=sum(x3[1])/m

for i in range(m):
    meanx[0]+=x3[i][0]
    meanx[1]+=x3[i][1]
meanx[0]/=m
meanx[1]/=m

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
# for y=1
x4=[]
x5=[]
# for y=0
x6=[]
x7=[]

# calculate u0 & u1
u0=np.array([0.,0.])
u1=np.array([0.,0.])
phi=0
m0=0
m1=0
for i in range(m):
    x2.append(np.array([x3[i][0],x3[i][1]]))

    if(y2[i]==1):
        x4.append(float(x3[i][0]))
        x5.append(float(x3[i][1]))
        u1+=x2[i]
        phi+=1
        m1+=1
    elif(y2[i]==0):
        x6.append(float(x3[i][0]))
        x7.append(float(x3[i][1]))
        u0+=x2[i]
        m0+=1
phi/=m
X=np.array(x2)
Y=np.array(y2)


u0/=m0
u1/=m1

print("u0=\n{}\n".format(u0))
print("u1=\n{}\n".format(u1))

# calculate Sigma
sigma=np.zeros((n,n))
for j in range(m):
    if(Y[j]==0):
        sigma+=np.outer(X[j]-u0,X[j]-u0)
    else:
        sigma+=np.outer(X[j]-u1,X[j]-u1)

sigma/=m
print("Sigma=\n{}\n".format(sigma))

sigmainv=np.linalg.inv(sigma)

# Q4(b)&(c)
plt.ion()
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)

# plots training data and linear separator
def pl():
    ax1.clear()
    ax1.scatter(x4, x5, label= "y='Canada' Training Data", color= "red",
                marker= ".", s=10)
    ax1.scatter(x6, x7, label= "y='Alaska' Training Data", color= "blue",
                marker= ".", s=10)


    xvalues=np.linspace(min(min(x4),min(x6)),max(max(x4),max(x6)),100)

    Xcoeff = np.matmul(sigmainv,(u0-u1)) + np.matmul((u0-u1).T,sigmainv)

    the=[0,0,0]

    # theta[0]=intercept = (constant in Xcoeff) + u1' * inv(sigma) * u1  -  u0' * inv(sigma) * u0  -  2log(phi/(1-phi))
    the[0]=np.matmul(u1.T,np.matmul(sigmainv,u1)) - np.matmul(u0.T,np.matmul(sigmainv,u0)) - 2*math.log((phi/(1-phi)))
    the[1]=Xcoeff[0]
    the[2]=Xcoeff[1]

    yvalues=(-the[1]*xvalues-the[0])/the[2]

    ax1.plot(xvalues, yvalues, label="Linear separator",color ="#66cc00")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Q4 (b)&(c)')
    plt.show()
    plt.pause(0.001)

# plt.ioff()
pl()

# Q4(d)

sigma0=np.zeros((n,n))
sigma1=np.zeros((n,n))
m0=0
m1=0
for j in range(m):
    if(Y[j]==0):
        sigma0+=np.outer(X[j]-u0,X[j]-u0)
        m0+=1
    else:
        sigma1+=np.outer(X[j]-u1,X[j]-u1)
        m1+=1

sigma0/=m0
sigma1/=m1

print("Sigma0=\n{}\n".format(sigma0))
print("Sigma1=\n{}\n".format(sigma1))

sigma0inv=np.linalg.inv(sigma0)
sigma1inv=np.linalg.inv(sigma1)

# constant = log(det(sigma1)/det(sigma0)) - 2*log(phi/(1-phi))
constant= math.log(abs(np.linalg.det(sigma1)/np.linalg.det(sigma0))) + 2*math.log(phi/(1-phi))

def findval(x,y):
    v = np.array([x,y])
    # (X-u1)' * inv(sigma1) * (X-u1)  -  (X-u0)' * inv(sigma0) * (X-u0)  +  constant
    ans=np.matmul((v-u1).T,np.matmul(sigma1inv,(v-u1))) - np.matmul((v-u0).T,np.matmul(sigma0inv,(v-u0))) + constant
    return ans

# Plots Quadratic Decision boundary as a contour
def plot_2():

    xvalues = np.linspace(min(min(x4),min(x6)),max(max(x4),max(x6)), 100)
    yvalues = np.linspace(min(min(x5),min(x7)),max(max(x5),max(x7)), 100)

    xvalues, yvalues = np.meshgrid(xvalues, yvalues)
    Zs = np.array([findval(x,y) for x,y in zip(np.ravel(xvalues), np.ravel(yvalues))])
    Z = Zs.reshape(xvalues.shape)

    CS=plt.contour(xvalues, yvalues, Z,[0])
    CS.collections[0].set_label("Quadratic boundary")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Q4 (e)')
    plt.show()
    plt.pause(0.001)

plt.ioff()
plot_2()
