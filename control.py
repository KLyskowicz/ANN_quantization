from SOM_Kohonen import SOM_Kohonen
from Neural_Gas import Neural_Gas
import os
import random
import math
import numpy as np

def make_group_of_points(data_amount, centers, radius):
    for_one_center = int(data_amount/len(centers))
    data = []
    for center in centers:
        for i in range(for_one_center):
            if i<for_one_center*0.8:
                data.append( [ random.uniform( center[0]-0.5*radius, center[0]+0.5*radius ), random.uniform( center[1]-0.5*radius, center[1]+0.5*radius ) ] )
            else:
                data.append( [ random.uniform( center[0]-radius, center[0]+radius ), random.uniform( center[1]-radius, center[1]+radius ) ] )
    return data

Data_amount = 500
Input_amount = 2
Input_min_max_value = [[-10,12],[-10,10]]
Radius = 2

Neuron_amount = 30
Lr_max = 0.5
Lr_min = 0.01
Neighbourhood_max = 0.5
Neighbourhood_min = 0.01

if not os.path.exists('out'):
    os.makedirs('out')
Path = os.path.join(os.getcwd(), 'out')

# data from file
fromFile = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '\\' + 'data.csv', delimiter=',')
Data = fromFile[:,0:2]

# random data
# Data = make_group_of_points(Data_amount, Centers, Radius)
# Data = []
# x = 10
# for i in range(int(len(data)/x)):
#     Data.append(data[i*x])

mapa = SOM_Kohonen(Input_amount, Neuron_amount, Lr_min, Lr_max, Neighbourhood_min, Neighbourhood_max, Input_min_max_value)
mapa.learn(Data, Path, 'Kohonen')

# mapa = Neural_Gas(Input_amount, Neuron_amount, Lr_min, Lr_max, Neighbourhood_min, Neighbourhood_max, Input_min_max_value)
# mapa.learn(Data, Path, Name)
