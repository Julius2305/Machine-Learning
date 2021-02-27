import numpy as np
import time
import os
from matplotlib import pyplot as plt
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import cv2
#from scipy.optimize import curve_fit

#erste auslesung der Daten
f = open("D:\\Studium\\Bachelorarbeit\\Machine Learning\\loss_function\\complete_statistics_changing_gapweight_clusterweight0,5.txt", "r")
# Using readlines()
Lines = f.readlines()
f.close()
array = []

for line in Lines:
    if "min_distance" in line:
        #tmp_array = line.split("number_of_pixels: tensor(")
        tmp_array = line.split("min_distance: ")
        tmp_array2 = tmp_array[1].split(",")
        tmp_array2 = tmp_array2[0].split(".")
        if "tensor" in tmp_array2[0]:
            tmp_array2 = tmp_array2[0].split("tensor(")
            tmp_array2 = tmp_array2[1].split(")")
        array.append(float(tmp_array2[0]))
        #print(tmp_array2[0])
array1 = np.array(array)

#zweite auslesung der Daten
f = open("D:\\Studium\\Bachelorarbeit\\Machine Learning\\loss_function\\complete_statistics_changing_gapweight_clusterweight1,1.txt", "r")
Lines = f.readlines()
f.close()
array = []

for line in Lines:
    if "min_distance" in line:
        #tmp_array = line.split("number_of_pixels: tensor(")
        tmp_array = line.split("min_distance: ")
        tmp_array2 = tmp_array[1].split(",")
        tmp_array2 = tmp_array2[0].split(".")
        if "tensor" in tmp_array2[0]:
            tmp_array2 = tmp_array2[0].split("tensor(")
            tmp_array2 = tmp_array2[1].split(")")
        array.append(float(tmp_array2[0]))
        #print(tmp_array2[0])
array2 = np.array(array)

#dritte auslesung der Daten
f = open("D:\\Studium\\Bachelorarbeit\\Machine Learning\\loss_function\\complete_statistics_changing_gapweight_clusterweight2.txt", "r")
Lines = f.readlines()
f.close()
array = []

for line in Lines:
    if "min_distance" in line:
        #tmp_array = line.split("number_of_pixels: tensor(")
        tmp_array = line.split("min_distance: ")
        tmp_array2 = tmp_array[1].split(",")
        tmp_array2 = tmp_array2[0].split(".")
        if "tensor" in tmp_array2[0]:
            tmp_array2 = tmp_array2[0].split("tensor(")
            tmp_array2 = tmp_array2[1].split(")")
        array.append(float(tmp_array2[0]))
        #print(tmp_array2[0])
array3 = np.array(array)








#x = np.arange(0, 10, 0.1)
#arr = np.array([1, 2, 3, 4, 5])


x = np.arange(0, 10, 0.1)
x2 = np.arange(0, 5000, 50)

#changing clusterzize weight
# plt.plot(x, array1, label="Minimal cluster distance with Gapweight 1000")
# plt.plot(x, array2, label="Minimal cluster distance with Gapweight 3000")
# plt.plot(x, array3, label="Minimal cluster distance with Gapweight 5000")

#changing gap-weight
#plt.plot(x2, array1, label="Minimal cluster distance with Clusterweight 0.5")
#plt.plot(x2, array2, label="Minimal cluster distance with Clusterweight 1.1")
plt.plot(x2, array3, label="Minimal cluster distance with Clusterweight 2")


plt.xlabel("Weight of the gap-loss")
plt.ylabel("Minimal distance(in Pixels)")

plt.legend(loc="upper left")
plt.grid()
plt.ylim(-1, 10)
plt.axhline(color='black', lw=0.75)
plt.axvline(color='black', lw=0.75)
plt.savefig("min_distance_changing_gapweight_clusterweight2")
plt.show()
