'''Двомірне відображення v i m.
Від зміни щвидкості навчання та параметра бета2'''

import numpy as np
import array as arr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# оголошення за задання необхідних змінних
hiddenSize =28 # кількість нейронів на прихованому шарі
alpha = 0.01 # швидкість навчання (коефіцієнт альфа)
eps = 0.000001 # бажана точнсть навчання
num =3# кількість шарів
beta1=0.9
beta2=0.999
arr_eps = [] # масив для значень похибки навчання
arr_age = [] # масив для значень кількості епох навчання

# задання тренувального набору даних


#print(x , len(x[0]))
'''for i in range(2):
    for value in xx[i].values():
        x=list(NumEp.values())+list(xx[i].values())
        print('testov_masuv_X=', x)

    for value in yy[i].values():
        y=list(NumEpY.values())+list(yy[i].values())
        print('testov_masuv_Y=', y)'''
        

            

def sigmoid(x): # функція активації
    c=1
    return 1 / (1 + np.exp(-x*c))
    #return 1+0.5*np.arctan(x)
    
    #return c*x
    #return (np.exp(c*x)-np.exp(-c*x)) / (np.exp(c*x) + np.exp(-c*x))
def sigmoid_output_to_derivative(output): # метод обчислення похідної від функції активації
    c=1
    #return c
    return output*c*(1 - output)
    #return 1 - output**2
def gen_synapse(x, y, hiddenSize, num): # генерація початкових ваг
    synapse = []
    np.random.seed(1)

    for i in range(num):
        if i == 0:
            synapse.append(2 * np.random.random((len(x[0]),hiddenSize)) - 1)
        elif i == num - 1:
            synapse.append(2 * np.random.random((hiddenSize,len(y[0]))) - 1)
        else:
            synapse.append(2 * np.random.random((hiddenSize,hiddenSize)) - 1)
    return synapse
    


Numt1=[0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Numt2=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
Numt3=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

Num81=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num82=[1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num83=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,0]
Num84=[1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num85=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1]

Num91=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num92=[1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num93=[1,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num94=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0]
Num95=[1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]

Num01=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num02=[1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num03=[1,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num04=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1]
Num05=[1,1,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]

Num21=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num22=[1,1,1,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num23=[1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num24=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0]
Num25=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1]

Num31=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num32=[1,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num33=[1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num34=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1]
Num35=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]

Num41=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num42=[0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num43=[1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num44=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0]
Num45=[1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1]

Num61=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num62=[0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num63=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0]
Num64=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1]
Num65=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1]

Num71=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num72=[0,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num73=[1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num74=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]
Num75=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0]


Num11=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num12=[0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num13=[0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num14=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0]
Num15=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1]

Num51=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num52=[1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num53=[0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num54=[1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num55=[1,1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
#Num56=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1]
#Num57=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]
             
NumEpt= [ Numt1, Numt2, Numt3]
Num0X=[Num01,Num02, Num03,Num04,Num05]
Num2X=[Num21,Num22, Num23,Num24,Num25]
Num3X=[Num31,Num32, Num33,Num34,Num35]
Num4X=[Num41,Num42, Num43,Num44,Num45]
Num6X=[Num61,Num62, Num63,Num64,Num65]
Num7X=[Num71,Num72, Num73,Num74,Num75]
Num8X=[Num81,Num82, Num83,Num84,Num85]
Num9X=[Num91,Num92, Num93,Num94,Num95]          
Num1X=[Num11,Num12, Num13,Num14,Num15]
Num5X=[Num51,Num52,Num53, Num54, Num55]
XX=[Num0X,Num1X,Num2X,Num3X,Num4X,Num5X,Num6X,Num7X,Num8X,Num9X]

NumEpY=[[0],[0],[0]]
Num0Y=[[1],[1],[1],[1],[1]]
Num1Y=[[1],[1],[1],[1],[1]]
Num2Y=[[1],[1],[1],[1],[1]]
Num3Y=[[1],[1],[1],[1],[1]]
Num4Y=[[1],[1],[1],[1],[1]]
Num5Y=[[1],[1],[1],[1],[1]]
Num6Y=[[1],[1],[1],[1],[1]]
Num7Y=[[1],[1],[1],[1],[1]]
Num8Y=[[1],[1],[1],[1],[1]]
Num9Y=[[1],[1],[1],[1],[1]]
YY=[Num0Y,Num1Y,Num2Y,Num3Y,Num4Y,Num5Y,Num6Y,Num7Y,Num8Y,Num9Y]
X00=[];X0=X00+NumEpt
for znaX0 in XX[0]: X0.append(znaX0)

X11=[];X1=X11+NumEpt
for znaX1 in XX[1]: X1.append(znaX1)

X22=[];X2=X22+NumEpt
for znaX2 in XX[2]: X2.append(znaX2)

X33=[];X3=X33+NumEpt
for znaX3 in XX[3]: X3.append(znaX3)

X44=[];X4=X44+NumEpt
for znaX4 in XX[4]: X4.append(znaX4)

X55=[];X5=X55+NumEpt
for znaX5 in XX[5]: X5.append(znaX5)

X66=[];X6=X66+NumEpt
for znaX6 in XX[6]: X6.append(znaX6)

X77=[];X7=X77+NumEpt
for znaX7 in XX[7]: X7.append(znaX7)

X88=[];X8=X88+NumEpt
for znaX8 in XX[8]: X8.append(znaX8)

X99=[];X9=X99+NumEpt
for znaX9 in XX[9]: X9.append(znaX9)

xx=np.array([X0,X1,X2,X3,X4,X5,X6,X7,X8,X9])
#print('testov_masuv_X=',xx, len(xx[0]),len(xx),type(xx))

Y00=[];Y0=Y00+NumEpY
for znaY0 in YY[0]: Y0.append(znaY0)

Y11=[];Y1=Y11+NumEpY
for znaY1 in YY[1]: Y1.append(znaY1)

Y22=[];Y2=Y22+NumEpY
for znaY2 in YY[2]: Y2.append(znaY2)

Y33=[];Y3=Y33+NumEpY
for znaY3 in YY[3]: Y3.append(znaY3)

Y44=[];Y4=Y44+NumEpY
for znaY4 in YY[4]: Y4.append(znaY4)

#print(Y1)
Y55=[];Y5=Y55+NumEpY
for znaY5 in YY[5]: Y5.append(znaY5)

Y66=[];Y6=Y66+NumEpY
for znaY6 in YY[6]: Y6.append(znaY6)

Y77=[];Y7=Y77+NumEpY
for znaY7 in YY[7]: Y7.append(znaY7)

Y88=[];Y8=Y88+NumEpY
for znaY8 in YY[8]: Y8.append(znaY8)

Y99=[];Y9=Y99+NumEpY
for znaY9 in YY[9]: Y9.append(znaY9)
#print(Y5)
yy=np.array([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9])
#print('testov_masuv_Y=', yy,type(yy))
nn=len(X0)
mm=len(xx)
print('цифр=',mm, 'krok=0.0000001','iter=100',yy.shape, xx.shape )
kilkist=arr.array('d',[])
chastota=arr.array('d',[])   
delta_n=arr.array('d',[])              
delta=arr.array('d',[])
delta_pox=arr.array('d',[])
XX=arr.array('d',[])
xxx=arr.array('d',[])
arr_age=arr.array('d',[])
arr_eps=arr.array('d',[])
alpha_x=arr.array('d',[])
epsilon=arr.array('d',[])
layer_errors=arr.array('d',[])
kilkist_alpha=arr.array('d',[])
ages=10
crok=0.05
e=0.00001
px=arr.array('d',[])
py=arr.array('d',[])
pc=[]
def Cart(beta2,c0):
    
    D=[]
    X=arr.array('d',[])
    Y=arr.array('d',[])
    for j in range(len(yy)):
        #print(len(yy))
        x=np.array(xx[j])
        y=np.array(yy[j])
        #print(len(y), len(x), type(y))
        m=[]
        v=[]
        vhat=[]
       
        def training(x, y, alpha, eps, hiddenSize, synapse, num, arr_age, arr_eps, alpha_x, chastota, xx, delta_n, xxx, epsilon, kilkist,kilkist_alpha, beta1, beta2): # метод навчання нейронної мережі
               
            delta=[]
            #print("Навчання нейронної мережі")
            age = 1
           
            while True:
                age += 1
                layers = []
                for i in range (num + 1):
                    if i == 0:
                        layers.append(x)
                    else:
                        layers.append(sigmoid(np.dot(layers[i - 1],synapse[i - 1])))

                layer_errors = []
                layer_deltas = []
                
                
                layer_errors.append(layers[num] - y)

                e = np.mean(np.abs(layer_errors[0]))

                if (age % 1) == 0:
                    arr_age.append(age)
                    arr_eps.append(e)
                    alpha_x.append(alpha)
                    #print("Похибка на " + str(age) + " ітерації: " + str(e))'''

                if(age >ages):
                    break

                if (e < eps):
                    #print("Точність " + str(round(e, 4)) + " досягнута за " + str(age) + " епох(и)")
                    break

                layer_deltas.append(layer_errors[0] * sigmoid_output_to_derivative(layers[num]))              
                #d=layer_errors[0]
                m=[0.0]
                v=[0.0]
                vhat=[0.0]
                m.append( beta1/(age+1) * m[0] + (1.0 - beta1/(age+1)) * layer_deltas[0])
                v.append((beta2 * v[0]) + (1.0 - beta2) * layer_deltas[0]**2)
                vhat=np.array(v[1])
                d=m[1]/np.sqrt(vhat+ 1e-8)
                #print('d=',len(d))
                d.shape=(8,1)
                #print('d=',d)
                d=[d]* int(hiddenSize)
                #d0.shape(5,8)
                #print('d0',len(d), 'len(m)',len(m))
                for i in range (num - 1):
                    layer_errors.append(layer_deltas[i].dot(synapse[num - 1 - i].T))
                    layer_deltas.append(layer_errors[i + 1] * sigmoid_output_to_derivative(layers[num - 1 - i]))
                    layer_deltass=layer_errors[i + 1] * sigmoid_output_to_derivative(layers[num - 1 - i])
                    #d[i+1]=layer_deltas[i]*((d[i]*1.0001)-(((d[i]*(layer_deltas[i+1]-layer_deltas[i])))*((d[i]*(layer_deltas[i+1]-layer_deltas[i]))))/(((layer_deltas[i+1]-layer_deltas[i]))*(d[i]*(layer_deltas[i+1]-layer_deltas[i]))))
                    # m(t) = beta1(t) * m(t-1) + (1 - beta1(t)) * g(t)
                    #print('num=', num,'i=',i,'i-1=', i-1, 'i+1=',i+1, len(m[i-1]), len (layer_deltas[i]))
                    #print('len(layer_deltass)',len(layer_deltass), 'len(m)', len(m[i-1]),i-1)
                    m = beta1**(age+1) * m[i-1] + (1.0 - beta1**(age+1)) * layer_deltass
                    #print( 'len(m)', len(m))
                    # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                    v = (beta2 * v[i-1]) + (1.0 - beta2) * layer_deltass**2
                    #print('v=',v)
                    
                    # vhat(t) = max(vhat(t-1), v(t))
                    vhat = max(max(vhat.reshape(-1,1)), max(v.reshape(-1,1)))
                    #print(len(vhat),vhat,len(v))
                    dd= m / (np.sqrt(vhat)+ 1e-8)
                    # x(t) = x(t-1) - alpha(t) * m(t) / sqrt(vhat(t)))
                    #d[i+1] = (beta1**(age+1) * m[i-1] + (1.0 - beta1**(age+1)) * layer_deltas[i]) / (np.sqrt((max(np.array((beta2 * v[i-1]) + (1.0 - beta2) * layer_deltas[i]**2).reshape(-1,1))) + 1e-8)) 
                    #print('dd',dd)
                    d.append(dd)                                    
                    #print('layer_errors=',layer_errors)    
                for i in range (num):
                    #print('len(layers[num - 1 - i])',len(layers[num - 1 - i]), 'len(d[i])',len(d[i]))
                    synapse[num - 1 - i] -= alpha * (layers[num - 1 - i].T.dot(d[i]))
                    #print('np.mean(synapse[num - 1 - i])'+str(synapse[num - 1 - i])+'='+str(np.mean(synapse[num - 1 ])))
                
                    delta.append(np.mean(synapse[num - 1]))
                    
                    '''#print('delta',delta)
                    #print('len(synapse[num - 1 - i])',len(synapse[num - 1 - i]))
                #delta_pox.append(np.mean(synapse[num - 1]))
                #print('delta_pox',delta_pox)
                chastota.append(alpha-delta[age-1]-delta[age-1]*delta[age-1])
                #XX.append(delta[age])
            #print(m)
            #delta_pox.append(np.mean(synapse[num - 1]))    
            kilkist_alpha.append(alpha)
            #kilkist_chastota.append(np.log(len(xx)/alpha))
            #print('min(delta),max(delta)=', min(delta),max(delta))'''
            '''A=synapse[num - 1]
            for i in range(hiddenSize):
                delta_pox.append(A[i])'''
            '''data=(min(delta)+max(delta))/2
            for delta0 in delta:
                delta_pox.append(delta0)
                delta0=data
                delta0=alpha-delta0-delta0*delta0    
                delta_n.append(delta0)
                #print(len(delta_n))
                xxx.append(alpha)
                epsilon.append(e)
                #print('data0',data0,alpha)
                data=delta0
            kilkist.append((len(delta_n)))
            #print(kilkist)'''
            mm=len(m)
            for i in range(mm):
                for ii in range(hiddenSize):
                     X.append(m[i][ii])
                     Y.append(v[i][ii])   
                     #print('Y',Y)
            
        synapse = gen_synapse(x, y, hiddenSize, num)        
        training(x, y, alpha, eps, hiddenSize, synapse, num, arr_age, arr_eps, alpha_x, chastota, XX, delta_n, xxx, epsilon, kilkist,kilkist_alpha, beta1, beta2)
    #D=np.array(D)    
    X=np.array(X)
    Y=np.array(Y)
    nn=len(X) 
    #print(len(X),len(Y), max(X),min(X),max(Y),min(Y))
    
    X.shape=10,8*28
    #X.reshape(10,8*28)
    Y.shape=10,8*28
    #Y.reshape(10,8*28)
    for i in range(10-1):  
        for ii in range(8*28-1):
        #print('D', len(D))
            if(abs(X[i][ii+1])>0.01):
                c='white'
                return(c) 
                break
            D.append((X[i][ii+1]+1)**2+(Y[i][ii+1]+1)**2)
        #print('D', D)
        c='black'
        if(abs(D[7]-D[6])<e): c='red'; return(c)
        if(abs(D[7]-D[5])<e): c='orange'; return(c)
        if(abs(D[7]-D[4])<e): c='yellow'; return(c)
        if(abs(D[7]-D[3])<e): c='green'; return(c)
        if(abs(D[7]-D[2])<e): c='cyan'; return(c)
        if(abs(D[7]-D[1])<e): c='blue'; return(c)
        if(abs(D[7]-D[0])<e): c='violet'; return(c)
        else: return(c)
for beta2 in np.arange(0.1,0.99,crok):
    print('beta2',round(beta2,6), 'цифра=',mm)
    for c0 in np.arange(0.0,0.5,crok):
        c=Cart(beta2,c0)
        px.append(beta2)
        py.append(c0)
        pc.append(c)
        #print(pc)
plt.scatter(py,px,marker=',',s=0.2,color=pc)
plt.xlabel('$a$',fontsize=14)
plt.ylabel('$b$',fontsize=14)
plt.savefig("Car_dinamic_v_m_B1_0.9_B2_0.99_0.9999_c0_0_0.005_e_0.00001.png",dpi=300)
plt.show()    
    
    
    
'''   
delta_n=np.array(delta_n)
#print('len(delta_n)',len(delta_n))
delta=np.array(delta)
delta_pox=np.array(delta_pox)
kilkist=np.array(kilkist)
arr_age=np.array(arr_age)
arr_eps=np.array(arr_eps)
chastota=np.array(chastota)
alpha_x=np.array(alpha_x)
XX=np.array(xx)
xxx=np.array(xxx)
epsilon=np.array(epsilon)
layer_errors=np.array(layer_errors)
#x=np.arange(0.01,1.01,0.001)
kilkist_alpha=np.array(kilkist_alpha)
#print('epsilon',epsilon,'len(epsilon)',len(epsilon))
#Визначення спектру частот похибки
#та визначення оптимальної швидкості навчання
alpha=np.arange(ll,lll,crok)
nm=int(((len(arr_age)-len(kilkist_alpha))*num)/mm)
print('mm',mm,'nm',nm)
print('len(xxx)',len(xxx),'len(arr_age)',len(arr_age),'len(kilkist_alpha)',len(kilkist_alpha),nm, 'len(XX)',len(XX),'len(chastota)',len(chastota))
print('len(delta_pox)',len(delta_pox), 'len(delta_n)', len(delta_n),'len(alpha)',len(alpha))

alpha=np.arange(ll,lll,crok)
delta_n.shape=mm,len(alpha),3*(ages-1)
delta_pox.shape=mm,len(alpha),3*(ages-1)
#delta_pox.shape=mm,len(alpha),3*(ages-1)
#np.save('delta_n.npy', delta_n)
xxx.shape=mm,nm
epsilon.shape=mm,nm
#print('delta_n',delta_n)
x=[]
y=arr.array('d',[])
z=[]

for ii in range(mm):
    datas =delta_pox[ii]
    for jj in range(len(alpha)):
        data=datas[jj]
        #print(ii,'похибка=',round(np.abs(np.mean(delta_pox[ii])),6))
        ps =np.abs(np.fft.fft(data))
        time_step =1.0
        freqs = np.fft.fftfreq(data.size, time_step)
        idx = np.argsort(freqs)
   
        garm=max(ps[int(nm/8):int(7*nm/8)])
        mm=list(ps).index(garm)
        #print ('max_ps',garm)
        if epsilon[ii][mm]<0.1:
            alpha_optumalne=xxx[ii][mm]
            epsilon_min=epsilon[ii][mm]
            print( 'цифра=',ii,';', 'alpha оптимальне=',alpha_optumalne,';', 'мінімальна похибка=',round(epsilon_min, 5))
        
        else: 
            print('відсутність процесу навчання')
        #print (freqs[idx].shape,ps[idx].shape )            
        plt.plot(freqs[idx], ps[idx])
        plt.xlabel(u"ω")
        plt.ylabel(u"S(ω)")
        plt.show()

        #print(len(xxx),len(delta_n),len(kilkist), len(x))
        
        plt.scatter(xxx[ii], delta_pox[ii], s=1.0, alpha=0.9)
        plt.title("Діаграма розгалуження")
        plt.xlabel(u"alpha")
        plt.ylabel(u"x")
        plt.show()
        plt.scatter(xxx[ii], delta_n[ii], s=1.0, alpha=0.9)
        plt.title("Діаграма розгалуження")
        plt.xlabel(u"alpha")
        plt.ylabel(u"x")
        plt.show()
        
        #print('len(freqs[idx])',len(freqs[idx]),'len(ps[idx])',len(ps[idx]),len(xxx[ii]))
       for iii in range(len(freqs[idx])):
            if np.abs(freqs[idx][iii])<0.4:
                x.append(freqs[idx][iii])
                #print(alpha[jj],freqs[idx][iii])            
                z.append(ps[idx][iii])
                y.append(alpha[jj])
    from mpl_toolkits.mplot3d import Axes3D
      
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter( x,y,z, s=1.0,alpha=0.9, color='b')
    ax.set_xlabel(u"ω") 
    ax.set_ylabel(u"alpha") 
    ax.set_zlabel(u"S(ω)") 
    plt.grid(True)
    plt.show()
    
    plt.scatter(XX, chastota, s=1, alpha=0.9)
    plt.xlabel(u"xn")
    plt.ylabel(u"xn+1")
    plt.show() 
    
    plt.scatter(x, kilkist, s=1, alpha=0.9)
    plt.xlabel('Кількість епох (age)')
    plt.ylabel('Похибка навчання (eps)')
    plt.show()

    fig=plt.figure() 
    ax=Axes3D(fig)
    ax.scatter(arr_age, alpha_x, arr_eps, s=0.7, color='b')
    ax.set_xlabel('Кількість епох (age)') 
    ax.set_ylabel('Швидкість навчання (alpha)') 
    ax.set_zlabel('Похибка навчання (eps)') 
    plt.grid(True)
    plt.show()'''
    
'''
    fig=plt.figure() 
    ax=Axes3D(fig) 
    ps[idx].shape=len(alpha),28
    x=freqs[idx]
    x.shape=len(alpha),28
    y=alpha
    X,Y=np.meshgrid(x,y)
    z0=kk
    #print(X,Y,z0)
    surf=ax.plot_surface(X,Y,z0,rstride=1,cstride=1,linewidth=0,cmap=mpl.cm.hsv) 
    fig.colorbar(surf, shrink=0.10, aspect=50) 
    plt.grid(True)
    plt.xlabel(u"K")
    plt.ylabel(u"T")	
    plt.show()'''
    
    

'''
layers_num=arr.array('d',[])
def main(): # основний метод програми

    # оголошення за задання необхідних змінних
    hiddenSize = 28 # кількість нейронів на прихованому шарі
    alpha =1.031 # швидкість навчання (коефіцієнт альфа)
    eps = 0.000001 # бажана точнсть навчання
    num = 3 # кількість шарів
    arr_eps = [] # масив для значень похибки навчання
    arr_age = [] # масив для значень кількості епох навчання
    alpha_x = []
    chastota = []
    xx = []
    delta_n = []
    xxx = []
    epsilon = []
    kilkist = []
    kilkist_alpha=[]
    Numt1=[0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Numt2=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    Numt3=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    Num81=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
    Num82=[1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
    Num83=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,0]
    Num84=[1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
    Num85=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1]

    Num91=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
    Num92=[1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
    Num93=[1,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
    Num94=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0]
    Num95=[1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]

    Num01=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
    Num02=[1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
    Num03=[1,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
    Num04=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1]
    Num05=[1,1,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]

    Num21=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
    Num22=[1,1,1,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
    Num23=[1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
    Num24=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0]
    Num25=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1]

    Num31=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num32=[1,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num33=[1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num34=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1]
    Num35=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]

    Num41=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num42=[0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num43=[1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num44=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0]
    Num45=[1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
    
    Num61=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
    Num62=[0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
    Num63=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0]
    Num64=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1]
    Num65=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1]

    Num71=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
    Num72=[0,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
    Num73=[1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
    Num74=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]
    Num75=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0]
    
    
    Num11=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num12=[0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num13=[0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
    Num14=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0]
    Num15=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1]

    Num51=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num52=[1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num53=[0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num54=[1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    Num55=[1,1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
    #Num56=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1]
    #Num57=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]
             
    NumEpt= [ Numt1, Numt2, Numt2]
    Num0X=[Num01,Num02, Num03,Num04,Num05]
    Num2X=[Num21,Num22, Num23,Num24,Num25]
    Num3X=[Num31,Num32, Num33,Num34,Num35]
    Num4X=[Num41,Num42, Num43,Num44,Num45]
    Num6X=[Num61,Num62, Num63,Num64,Num65]
    Num7X=[Num71,Num72, Num73,Num74,Num75]
    Num8X=[Num81,Num82, Num83,Num84,Num85]
    Num9X=[Num91,Num92, Num93,Num94,Num95]          
    Num1X=[Num11,Num12, Num13,Num14,Num15]
    Num5X=[Num51,Num52,Num53, Num54, Num55]
    XX=[Num0X,Num1X,Num2X,Num3X,Num4X,Num5X,Num6X,Num7X,Num8X,Num9X]

    NumEpY=[[0],[0],[0]]
    Num0Y=[[1],[1],[1],[1],[1]]
    Num1Y=[[1],[1],[1],[1],[1]]
    Num2Y=[[1],[1],[1],[1],[1]]
    Num3Y=[[1],[1],[1],[1],[1]]
    Num4Y=[[1],[1],[1],[1],[1]]
    Num5Y=[[1],[1],[1],[1],[1]]
    Num6Y=[[1],[1],[1],[1],[1]]
    Num7Y=[[1],[1],[1],[1],[1]]
    Num8Y=[[1],[1],[1],[1],[1]]
    Num9Y=[[1],[1],[1],[1],[1]]
    YY=[Num0Y,Num1Y,Num2Y,Num3Y,Num4Y,Num5Y,Num6Y,Num7Y,Num8Y,Num9Y]
    X00=[];X0=X00+NumEpt
    for znaX0 in XX[0]: X0.append(znaX0)

    X11=[];X1=X11+NumEpt
    for znaX1 in XX[1]: X1.append(znaX1)

    X22=[];X2=X22+NumEpt
    for znaX2 in XX[2]: X2.append(znaX2)

    X33=[];X3=X33+NumEpt
    for znaX3 in XX[3]: X3.append(znaX3)

    X44=[];X4=X44+NumEpt
    for znaX4 in XX[4]: X4.append(znaX4)

    X55=[];X5=X55+NumEpt
    for znaX5 in XX[5]: X5.append(znaX5)

    X66=[];X6=X66+NumEpt
    for znaX6 in XX[6]: X6.append(znaX6)

    X77=[];X7=X77+NumEpt
    for znaX7 in XX[7]: X7.append(znaX7)

    X88=[];X8=X88+NumEpt
    for znaX8 in XX[8]: X8.append(znaX8)

    X99=[];X9=X99+NumEpt
    for znaX9 in XX[9]: X9.append(znaX9)

    xx=np.array([X0,X1,X2,X3,X4,X5,X6,X7,X8,X9])
    #print('testov_masuv_X=',xx, len(xx[0]),len(xx),type(xx))

    Y00=[];Y0=Y00+NumEpY
    for znaY0 in YY[0]: Y0.append(znaY0)

    Y11=[];Y1=Y11+NumEpY
    for znaY1 in YY[1]: Y1.append(znaY1)

    Y22=[];Y2=Y22+NumEpY
    for znaY2 in YY[2]: Y2.append(znaY2)

    Y33=[];Y3=Y33+NumEpY
    for znaY3 in YY[3]: Y3.append(znaY3)

    Y44=[];Y4=Y44+NumEpY
    for znaY4 in YY[4]: Y4.append(znaY4)

    #print(Y1)
    Y55=[];Y5=Y55+NumEpY
    for znaY5 in YY[5]: Y5.append(znaY5)

    Y66=[];Y6=Y66+NumEpY
    for znaY6 in YY[6]: Y6.append(znaY6)

    Y77=[];Y7=Y77+NumEpY
    for znaY7 in YY[7]: Y7.append(znaY7)

    Y88=[];Y8=Y88+NumEpY
    for znaY8 in YY[8]: Y8.append(znaY8)

    Y99=[];Y9=Y99+NumEpY
    for znaY9 in YY[9]: Y9.append(znaY9)
    #print(Y5)
    yy=np.array([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9])
    #print('testov_masuv_Y=', yy,type(yy))
    for j in range(len(yy)):
    
        x=np.array(xx[j])
        y=np.array(yy[j])
        synapse = gen_synapse(x, y, hiddenSize, num)

        # навчання нейроної мережі
        training(x, y, alpha, eps, hiddenSize, synapse, num, arr_age, arr_eps, alpha_x, chastota, xx, delta_n, xxx, epsilon, kilkist,kilkist_alpha)
        # перевірка роботи нейромережі на тренувальному наборі даних
        print("\nПеревірка роботи нейромережі на навчальних даних:")
        print(x)

        layers = []
        for i in range (num + 1): # прогонка даних по шарам нейромережі
            if i == 0:
                layers.append(x)
            else:
                layers.append(sigmoid(np.dot(layers[i - 1],synapse[i - 1])))
                
        print("\nПрогнозоване значення: ") # вивід результатів роботи нейронної мережі
        print(layers[num])
        #test1, test2, test3 = map(int, input("\nВведіть тестовий набір даних: ").split()) # введення тестового набору даних
        
        testSetX = [0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1] # задання введеного тестового набору даних

        layers = []
        for i in range (num + 1): # прогонка даних по шарам нейромережі
            if i == 0:
                layers.append(testSetX)
            else:
                layers.append(sigmoid(np.dot(layers[i - 1],synapse[i - 1])))
            
        print("\nПрогнозоване значення: ") # вивід результатів роботи нейронної мережі
        print(layers[num])
        layers_num.append(layers[num])
    if max(layers_num)> 0.9:
        print(layers_num,'\n', 'число =',layers_num.index(max(layers_num)))
    else:
        print('Це не цифра')
main() # виклик головного методу програми - main'''