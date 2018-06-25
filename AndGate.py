import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

x = T.vector('x') 
w = T.vector('w') 
b = T.scalar('b') 

z = T.dot(x,w)+b
y = ifelse(T.lt(z,0),0,1) 

neuron = theano.function([x,w,b],y)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

weights = [ 1, 1]
bias = -1.5

for i in range(len(inputs)):
    temp = inputs[i]
    output = neuron(temp,weights,bias)
    print('input 是 [x1,x2] = [%d,%d] \noutput 為 %d\n'%(temp[0],temp[1],output))
    
print('輸入一組test data為 [0,1]')
temp = [0,1]
output = neuron(temp,weights,bias)
print('結果為:%d'%(output))