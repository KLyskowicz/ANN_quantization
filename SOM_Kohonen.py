import numpy as np
import math 
import os
from Neuron import Neuron
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animation

class SOM_Kohonen(object):
    def __init__(self, input_amount, neuron_amount, lr_min, lr_max, neighbourhood_min, neighbourhood_max, input_min_max_value=[[-1.1,1.1],[-1.1,1.1]], potential_min=0.75):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr = 0
        self.neighbourhood_min = neighbourhood_min
        self.neighbourhood_max = neighbourhood_max
        self.neighbourhood = 0
        self.potential_min = potential_min
        self.activity_tab = []
        self.input_max_min = input_min_max_value
        self.dataX = []
        self.dataY = []
        self.neurons = [Neuron(input_amount, input_min_max_value, i) for i in range(neuron_amount)]

    def find_winner(self, one_input):
        distance = []
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                distance.append(neuron.distance_measure(one_input))
            else:
                distance.append(999999)
        return np.argmin(distance, axis=0)

    def lr_neighbourhood_update(self, step_number, steps_amount):
        self.lr = self.lr_max*pow(self.lr_min/self.lr_max, (step_number)/(steps_amount))
        self.neighbourhood = self.neighbourhood_max*pow(self.neighbourhood_min/self.neighbourhood_max, (step_number)/(steps_amount))

    def gauss(self, neuron_number, winner_number):
        return math.exp( -pow( self.neurons[winner_number].distance_measure(self.neurons[neuron_number].weights),2 ) / ( 2*pow(self.neighbourhood,2) ) )

    def weight_update(self, winner_number, one_input):
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                if neuron.number != winner_number:
                    gauss = self.gauss(neuron.number, winner_number)
                    new_weight = []
                    for weight, one_in in zip(neuron.weights, one_input):
                        new_weight.append(weight + self.lr * gauss * (one_in - weight))
                    neuron.update_weight(new_weight)
        new_weight = []
        for weight, one_in in zip(self.neurons[winner_number].weights, one_input):
            new_weight.append(weight + self.lr * (one_in - weight))
        self.neurons[winner_number].update_weight(new_weight)

    def dead_update(self, winner_number):
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                neuron.potential_update(self.potential_min, winner_number, len(self.neurons))
            else:
                neuron.potential = 1

    def dead_check(self):
        self.activity_tab.clear()
        for neuron in self.neurons:
            if neuron.potential < self.potential_min:
                self.activity_tab.append(0)
            else:
                self.activity_tab.append(1)

    def error_measure(self, data):
        error = 0
        for one_data in data:
            distance = self.neurons[self.find_winner(one_data)].distance_measure(one_data)
            error += distance*distance
        return error/len(data)

    def learn(self, data, path, name):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)

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
            for epo in range(100):
                data_number = frame*100+epo
                self.lr_neighbourhood_update(data_number, len(data))
                po_ilu_wylaczyc = 300
                if frame<po_ilu_wylaczyc:
                    self.dead_check()
                elif frame==po_ilu_wylaczyc:
                    self.activity_tab.clear
                    for i in range(len(self.neurons)):
                        self.activity_tab.append(1)
                winner_number = self.find_winner(data[data_number])
                self.weight_update(winner_number, data[data_number])
                if frame<po_ilu_wylaczyc:
                    self.dead_update(winner_number)
            return ln

        ani = animation.FuncAnimation(fig, animate_points, frames=np.arange(0, 100).tolist(), init_func=init)
        name1 = os.path.join(path, 'K_' + str(name) + ".mp4")
        ani.save(name1, writer=writer)

        name2 = os.path.join(path, 'K_' + str(name) + ".txt")
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



                