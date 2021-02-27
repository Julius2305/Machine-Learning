import numpy as np
import time
import os
from matplotlib import pyplot as plt
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import cv2
#from scipy.optimize import curve_fit

def determine_startcluster_size(path):
    filepath = "D:\\Studium\\Bachelorarbeit\\Machine Learning\\loss_function\\" + path + ".txt"
    f = open(filepath, "r")
    # Using readlines()
    Lines = f.readlines()
    in_result = 0
    tmp_array2 = ''
    tmp_array3 = []
    # Strips the newline character
    for line in Lines:
        if(in_result == 1):
            if "]" in line:
                tmp_array = line.split("]")
                tmp_array2 += tmp_array[0]
                in_result = 0
                array = tmp_array2.split(".,")
                result = []
                for value in array:
                    if "\n" in value:
                        result.append(int(value.split("\n        ")[1]))
                    elif " " in value:
                        if "." in value:
                            result.append(int(value.split(" ")[1].split(".")[0]))
                        else:
                            result.append(int(value.split(" ")[1]))
                    else:
                        # if "." in value:
                        #     result.append(int(value.split(".")[0]))
                        # else:
                        result.append(int(value))
                #print(torch.tensor(result).view(10,10))
                tmp_array2 = ''


                #find the start_cluster_size
                result = torch.tensor(result, dtype=torch.float32).view(10,10)
                img = result.round()
                img = img.detach().numpy()
                img = np.array(img, dtype=np.uint8)
                num_labels, labels_im = cv2.connectedComponents(img)

                start_value = labels_im[4][6]
                end_value = labels_im[0][2]

                #remodel the matrix, so that distance_transform can be used
                start_cluster = []

                if(start_value != 0 and end_value != 0):
                    column_nr = len(labels_im)
                    row_nr = len(labels_im[0])
                    for i in range(0, column_nr):
                        for j in range(0, row_nr):
                            if (labels_im[i][j] == end_value):
                                labels_im[i][j] = 0
                            elif (labels_im[i][j] == 0):
                                labels_im[i][j] = end_value
                            elif (labels_im[i][j] == start_value):
                                start_cluster.append([i,j])

                            if(end_value == start_value):
                                if(labels_im[i][j] == 0):
                                    start_cluster.append([i,j])
                #print(len(start_cluster))
                tmp_array3.append(len(start_cluster))

            else:
                tmp_array2 += line
        # if "min_distance" in line:
        #     #tmp_array = line.split("number_of_pixels: tensor(")
        #     tmp_array = line.split("min_distance: ")
        #     tmp_array2 = tmp_array[1].split(",")
        #     tmp_array2 = tmp_array2[0].split(".")
        #     print(tmp_array2[0])
        if "result_rounded:" in line:
            #tmp_array = line.split("number_of_pixels: tensor(")
            tmp_array = line.split("result_rounded: tensor([")
            tmp_array2 = tmp_array[1]
            in_result = 1
    return tmp_array3

#erste auslesung der Daten
#array1 = determine_startcluster_size("complete_statistics_changing_gapweight_clusterweight0,5")
array1 = determine_startcluster_size("complete_statistics_changing_clusterweight_gapweight1000")

#zweite auslesung der Daten
#array2 = determine_startcluster_size("complete_statistics_changing_gapweight_clusterweight1,1")
array2 = determine_startcluster_size("complete_statistics_changing_clusterweight_gapweight3000")

#dritte auslesung der Daten
#array3 = determine_startcluster_size("complete_statistics_changing_gapweight_clusterweight2")
array3 = determine_startcluster_size("complete_statistics_changing_clusterweight_gapweight5000")


x = np.arange(0, 10, 0.1)
arr = np.array([1, 2, 3, 4, 5])


x = np.arange(0, 10, 0.1)
x2 = np.arange(0, 5000, 50)

#changing gapweight stats
#plt.plot(x2, array1, label="Size of the cluster around the startpoint with Clusterweight 0.5")
#plt.plot(x2, array2, label="Size of the cluster around the startpoint with Clusterweight 1.1")
#plt.plot(x2, array3, label="Size of the cluster around the startpoint with Clusterweight 2")

#changing clusterweight stats
plt.plot(x, array1, label="Size of the cluster around the startpoint with Gapweight 1000")
plt.plot(x, array2, label="Size of the cluster around the startpoint with Gapweight 3000")
plt.plot(x, array3, label="Size of the cluster around the startpoint with Gapweight 5000")


plt.xlabel("Weight of the clustersize-loss")
           # family='serif',
           # color='r',
           # weight='normal',
           # size = 16,
           # labelpad = 6)
plt.ylabel("Clustersize(Number of Pixels)")
plt.legend(loc="upper left")
plt.grid()
plt.ylim(-1, 50)
plt.axhline(color='black', lw=0.75)
plt.axvline(color='black', lw=0.75)
plt.savefig("startcluster_size_changing_clusterweight")
plt.show()
