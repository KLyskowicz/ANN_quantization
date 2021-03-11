import numpy as np
import math 
import os
from Neuron import Neuron
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animation

class Neural_Gas(object):
    def __init__(self, input_amount, neuron_amount, lr_min, lr_max, neighbourhood_min, neighbourhood_max, input_min_max_value=[[-1.1,1.1],[-1.1,1.1]], potential_min=0.75):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr = 0
        self.neighbourhood_min = neighbourhood_min
        self.neighbourhood_max = neighbourhood_max
        self.neighbourhood = 0
        self.potential_min = potential_min
        self.activity_tab = []
        self.order_tab = []
        self.input_max_min = input_min_max_value
        self.neurons = [Neuron(input_amount, input_min_max_value, i) for i in range(neuron_amount)]

    def find_winner(self, one_input):
        distance = []
        for neuron in self.neurons:
            distance.append(neuron.distance_measure(one_input))
        return np.argmin(distance, axis=0)

    def sort(self, one_input):
        distance = []
        for neuron in self.neurons:
            distance.append([neuron.distance_measure(one_input), neuron.number])
        self.order_tab = sorted(distance, key=lambda x: x[0])

    def lr_neighbourhood_update(self, step_number, steps_amount):
        self.lr = self.lr_max*pow(self.lr_min/self.lr_max, (step_number)/(steps_amount))
        self.neighbourhood = self.neighbourhood_max*pow(self.neighbourhood_min/self.neighbourhood_max, (step_number)/(steps_amount))

    def weight_change_factor(self, number_in_order):
        return math.exp( -number_in_order/self.neighbourhood )

    def weight_update(self, one_input):
        for i, tab in zip(range(len(self.order_tab)), self.order_tab):
            weight_change_factor = self.weight_change_factor(i)
            new_weight = []
            for weight, one_in in zip(self.neurons[tab[1]].weights, one_input):
                new_weight.append(weight + self.lr * weight_change_factor * (one_in - weight))
            self.neurons[tab[1]].update_weight(new_weight)

    def data_normalize_Łukasz(self, data):
        maximum_XY = np.amax(data, axis=0)
        factor = math.sqrt(maximum_XY[0]*maximum_XY[0]+maximum_XY[1]*maximum_XY[1])
        for one_data in data:
            one_data[0] = one_data[0]/factor
            one_data[1] = one_data[1]/factor

    def error_measure(self, data):
        error = 0
        for one_data in data:
            distance = self.neurons[self.find_winner(one_data)].distance_measure(one_data)
            error += distance*distance
        return error/len(data)

    def learn(self, data, path, name):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        np.random.shuffle(data)
        self.dataX = [one_data[0] for one_data in data]
        self.dataY = [one_data[1] for one_data in data]
        fig, ax1 = plt.subplots()
        ln, = plt.plot([], [], 'ro')
        def init():
            ax1.set_xlim(-15, 15)
            ax1.set_ylim(-15, 15)
            ax1.scatter(self.dataX, self.dataY, color='blue')
            return ln,
        def animate_points(frame):
            xs2 = [neuron.weights[0] for neuron in self.neurons]
            ys2 = [neuron.weights[1] for neuron in self.neurons]
            ln.set_data(xs2, ys2)
            self.lr_neighbourhood_update(frame, len(data))
            self.sort(data[frame])
            self.weight_update(data[frame])
            return ln
        ani = animation.FuncAnimation(fig, animate_points, frames=np.arange(0, len(data)).tolist(), init_func=init)
        name1 = os.path.join(path, 'G_' + str(name) + ".mp4")
        ani.save(name1, writer=writer)
        name2 = os.path.join(path, 'G_' + str(name) + ".txt")
        error_save = open(name2,"w+")
        error_save.write(str( round( self.error_measure(data), 6 ) ))
        error_save.close()

    def print(self, data, path, i):
        name = str(i) + '.png'
        xs1 = []
        ys1 = []
        for one in data:
            xs1.append(one[0])
            ys1.append(one[1])
        xs2 = []
        ys2 = []
        for neuron in self.neurons:
            xs2.append(neuron.weights[0])
            ys2.append(neuron.weights[1])
        plt.scatter(xs1, ys1, color='blue')
        plt.scatter(xs2, ys2, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim (-15,15)
        plt.ylim (-15,15)
        # plt.title( str( round( self.error_measure(data), 6 ) ) )
        plt.savefig(os.path.join(path, name))
        plt.close()