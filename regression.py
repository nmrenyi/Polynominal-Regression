# RY first machine learning task: Regression
# y = a0 + a1x1 + a2x2^2 + ... + anxn^n
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# TODO MEASURE THE REGRESSION to the loss?
# TODO SGD / MINI BATCH COMPARE BETWEEN DIFFERENT METHODS
# TODO OTHER x^n and logistic regression
# TODO convert into network and use pytorch
# TODO change the learning rate

degree = 2 # poly degree, i.e. n in Line2
w = np.random.randn(1, degree + 1) # initialize parameter w
w = w[0]
# bias = np.random.randint(-10, 11) # 
# data_x, data_y, coef = datasets.make_regression(n_samples = 500,n_features = 1,n_targets = 1, bias = bias,noise = 10, coef=True) # TODO checkout parametres of the dataset
# data_x = [x[0] for x in data_x]

def getNumber(f):
    s = f.readline()
    s = s.split(' ')
    s[-1] = s[-1].replace('\n', '')
    return [float(x) for x in s]
# with open('data.txt', 'r') as f:
#     coef, bias = getNumber(f)
#     data_x = getNumber(f)
#     data_y = getNumber(f)


# with open('data.txt', 'w') as f:
#     f.write(str(coef) + ' ' + str(bias) + '\n')
#     f.write(' '.join([str(x) for x in data_x]) + '\n')
#     f.write(' '.join([str(y) for y in data_y]) + '\n')

# data_x = [1.65, 2.65,4.5, 3.65, 5.65, 6.65, 7.65]
# data_y = [448.36, 446.4,443.5, 444.27,	444.29,	445.07,	447.17]

# x^3 + 2 x^2 + 3x + 4
data_x = [ -3.0000,-2.5000,-2.0000,-1.5000,-1.0000,-0.5000,0,0.5000,1.0000,1.5000,2.0000,2.5000,3.0000,3.5000,4.0000]
data_y = [-14.0000,-6.6250,-2.0000,0.6250,2.0000,2.8750,4.0000   , 6.1250 ,  10.0000   ,16.3750  , 26.0000 ,  39.6250  , 58.0000  , 81.8750 , 112.0000]


# data_x = [1., 2., 3., 4., 5.]
# data_y = [2., 4., 6., 8., 10.]

# data_x = [-1, 0, 1, 2, 3]
# data_y = [4, 1, 0, 1, 4]

# data_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# data_y = [ 1,8,27,64,125,216,343,512,729,1000]

data = list(zip(data_x, data_y))

epoch = 500
learning_rate = 0.01
loss = 100
mini_batch_size = 32
batches = [len(data_x), mini_batch_size, 1]
labels = ['BGD', 'MINI BATCH', 'SGD']
plt.figure()

print(f'epoch = {epoch}')
for num, batch_size in enumerate(batches):
    for i in range(epoch):
        # print(f'epoch = {i}')
        # batch_size = 1 # SGD
        # batch_size = 32 # mini-batch SGD
        # batch_size = len(data_x) # BGD
        delta_w = np.zeros_like(w)
        last_loss = loss
        loss = 0.0
        shuffle(data)
        mini_data = data[0:batch_size]
        
        for x, y in mini_data:
            v_x = np.logspace(0, degree, degree + 1, base = x)
            y_hat = np.dot(w, v_x)
            delta_w += (y_hat - y) * v_x / batch_size
            loss += 0.5 / batch_size * ((y_hat - y) ** 2)
        w -= learning_rate * delta_w
        
        print(f'loss = {loss}')
        print(f'w = {w}')
        print('')
    print(f'{labels[num]} loss = {loss}')
    print(f'{labels[num]} train w = {w}')
    print('')
    func = np.poly1d(np.flipud(w))
    res_x = np.linspace(-5, 5)
    res_y = func(res_x)
    plt.plot(res_x, res_y, label = labels[num])


# print(f'Training complete, final loss = {loss}, final w = {w}')
# print(f'real coef = {coef}, bias = {bias}')
# print(f'train coef = {w[1]}, bias = {w[0]}')

def get_average(data):
    sum = 0.0
    for i in data:
        sum += i
    return sum / len(data)
def get_least_square(data_x, data_y):
    x_bar = get_average(data_x)
    y_bar = get_average(data_y)
    up = 0.0
    down = 0.0
    for x, y in list(zip(data_x, data_y)):
        up += (x - x_bar) * (y - y_bar)
        down += (x - x_bar) ** 2
    coef = up / down
    bias = y_bar - coef * x_bar
    return coef, bias

# ls_coef, ls_bias = get_least_square(data_x, data_y)
# print(f'ls   coef = {ls_coef}, bias = {ls_bias}')



# func_ls = np.poly1d([ls_coef, ls_bias])
# ls_x = np.linspace(-5, 5)
# ls_y = func_ls(ls_x)

# func_real = np.poly1d([coef, bias])
# real_x = np.linspace(-5, 5)
# real_y = func_real(real_x)
# plt.plot(real_x, real_y, label = 'real')

# plt.plot(ls_x, ls_y, label = 'least square')
plt.scatter(data_x, data_y)

plt.legend()
plt.title(f'linear regression epoch = {epoch}, mini-batch size = {mini_batch_size}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('result.png',dpi=600,format='png')

# plt.show()

# from sklearn import datasets#引入数据集
# #构造的各种参数可以根据自己需要调整
# X,y=datasets.make_regression(n_samples=100,n_features=4,n_targets=1,noise=1)
# print(len(X))
# print(len(y))
# print(X)
# print(y)
# ###绘制构造的数据###
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(X,y)
# plt.show()
