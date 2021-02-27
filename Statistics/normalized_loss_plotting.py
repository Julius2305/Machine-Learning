import numpy as np
import time
import os
from matplotlib import pyplot as plt
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import cv2
#from scipy.optimize import curve_fit

class CustomLoss(nn.Module):
    def init(self):
        super(CustomLoss,self).__init__()

    def forward(self, result_given, points_given, weightmatrix_given):
        #variables for easier variation of the loss function
        weight = 20000
        gap_weight = 3000 #1200
        cluster_size_weight = 3
        weight_weight = 1.1  #0.5
        result_size = result_given.size()
        loss = torch.tensor([0], dtype=torch.float32, requires_grad = True)

        for i in range(0, result_size[0]):
            result = result_given[i, 0, 0:, 0:]
            weightmatrix = weightmatrix_given[i, 0, 0:, 0:]
            # print("result", result)
            # result_img = result.round().view(10,10)
            # result_imgplot = plt.matshow(result_img.detach().numpy())
            points = points_given[i]
            # if i%100 == 0:
            #     print(i)
            #loss = torch.tensor([0], dtype=torch.float32, requires_grad = True)


            manhattan_distance_start_end = self.estimate_manhattan_distance(points[0][0], points[0][1], points[1][0], points[1][1])

            # soa_cells = torch.tensor([0], dtype=torch.float32, requires_grad = True)                      #sum of all cells
            # soa_cells_inv = torch.tensor([0], dtype=torch.float32, requires_grad = True)                  #sum of all cells inverted
            # for row in result:
            #     for column in row:
            #         soa_cells += column
            #         soa_cells_inv += (1-column)
            soa_cells = sum(sum(result))
            soa_cells_inv = 100 - sum(sum(result))

            #print(result)
            #print(soa_cells)

            #set the start and endpoint to 1
            loss_start = torch.tensor([0], dtype=torch.float32, requires_grad = True)
            if(result[points[0][0]][points[0][1]].round() == 0 or result[points[1][0]][points[1][1]] == 0):
                loss_start = (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight
                loss_start = loss_start.view(1)

            #first compute the clusters
            img = result.round()
            img = img.detach().numpy()
            img = np.array(img, dtype=np.uint8)
            num_labels, labels_im = cv2.connectedComponents(img)
            # print(labels_im)

            # imgplot = plt.matshow(labels_im, cmap=plt.cm.gray)
            # imgplot = plt.matshow(labels_im)
            # plt.show()

            start_value = labels_im[points[0][0]][points[0][1]]
            end_value = labels_im[points[1][0]][points[1][1]]
            # print(start_value)
            # print(end_value)

            #remodel the matrix, so that distance_transform can be used
            start_cluster = []
            gap_loss = torch.tensor([0], dtype=torch.float32)

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

                # perform the distance transform
                labels_im = np.array(labels_im, dtype=np.uint8)
                dist_img = cv2.distanceTransform(labels_im, distanceType=cv2.DIST_L1, maskSize=3).astype(np.float32)
                #print(dist_img[points[0][0]][points[0][1]])

                #get the min distance to the end_cluster from the start_cluster
                min_distance = dist_img[points[0][0]][points[0][1]]
                for cell in start_cluster:
                    if dist_img[cell[0]][cell[1]] < min_distance:
                        min_distance = dist_img[cell[0]][cell[1]]
                #print(min_distance)

                gap_loss = min_distance * soa_cells_inv * gap_weight #* soa_cells_inv  # *(1-result[0][0]) funktioniert auch nicht -> Backward kann das nicht (alles in eine Zeile packen -> das + scheint alles kaputt zu machen???)
                #loss = torch.cat((loss, gap_loss), 0)
            else:
                # loss_start += (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight
                # loss_start = loss_start.view(1)
                gap_loss = (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight
                gap_loss = gap_loss.view(1)

            #loss = torch.cat((loss, loss_start), 0)

            # cluster_cells = torch.tensor([0], dtype=torch.float32)                      #sum of all cluster_cells
            # for cell in start_cluster:
            #     #cluster_cells += result[cell[0]][cell[1]]
            #     cluster_cells = torch.cat((cluster_cells.view(1), result[cell[0]][cell[1]].view(1)), 0)
            #     cluster_cells = sum(cluster_cells)


            cluster_size_penalty = torch.tensor([0], dtype=torch.float32, requires_grad = True)
            #cluster_size_penalty = soa_cells * cluster_size_weight * abs(manhattan_distance_start_end - len(start_cluster))              #alternativ: penalty for every 1 outsde the clusters, and a penalty depending on len(start_cluster)
            cluster_size_penalty = sum((result*weightmatrix).view(100)) * weight_weight * abs(manhattan_distance_start_end - len(start_cluster))
            #möglicherweise einfach cluster_size_weight von weightmatrix abhängig machen? Auf jeden Fall noch weightmatrix bei points runtersetzen, sonst wird das weight dort viel zu hoch
            # weight_loss = sum((result*weightmatrix).view(100)) * weight_weight

            #loss = loss_start + lonelyness_penalty + single_cell_penalty + cluster_size_penalty + gap_penalty + loss
            # print("loss_start: " , loss_start)
            # print("gap_loss: " , gap_loss)
            # print("cluster_size_penalty: " , cluster_size_penalty)

            #Concatenate the loss with the other losses from the batch
            loss = torch.cat((loss, loss_start + gap_loss + cluster_size_penalty), 0)
            #loss = loss_start + gap_loss + cluster_size_penalty

        #print(sum(loss))
        #print(loss_start.grad_fn)
        # loss_start.retain_grad()
        # gap_loss.retain_grad()
        # cluster_size_penalty.retain_grad()
        # return gap_loss

        #return the mean of all losses over the batch
        return sum(loss)/result_size[0]
        #return loss_start + cluster_size_penalty + gap_loss #+ weight_loss
        #return sum(loss)
        #return sum(loss), gap_loss, cluster_size_penalty

        #print(result[points[0][0]][points[0][1]])


    def estimate_manhattan_distance(self, start_x, start_y, end_x, end_y):
        return abs(end_x - start_x) + abs(end_y - start_y)




def determine_loss(path, batch_size, points, weightmatrix):
    filepath = "D:\\Studium\\Bachelorarbeit\\Machine Learning\\loss_function\\" + path + ".txt"
    f = open(filepath, "r")
    # Using readlines()
    Lines = f.readlines()
    in_result = 0
    criterion = CustomLoss()
    tmp_array2 = ''
    tmp_array3 = []
    # Strips the newline character
    for line in Lines:
        if(in_result == 1):
            if "]" in line:
                tmp_array = line.split("]")
                tmp_array2 += tmp_array[0]
                in_result = 0
                array = tmp_array2.split(",")
                result = []
                for value in array:
                    if "\n" in value:
                        result.append(float(value.split("\n        ")[1]))
                    elif " " in value:
                        if "." in value:
                            result.append(float(value.split(" ")[1].split(".")[0]))
                        else:
                            result.append(float(value.split(" ")[1]))
                    else:
                        # if "." in value:
                        #     result.append(int(value.split(".")[0]))
                        # else:
                        result.append(float(value))
                #print(torch.tensor(result).view(10,10))
                tmp_array2 = ''
                result = torch.tensor(result, dtype=torch.float32)
                loss = criterion(result_given=result.view(batch_size,1,10,10), points_given=points, weightmatrix_given=weightmatrix)

                loss = loss.view(1)
                loss = loss.detach().numpy()
                tmp_array3.append(loss[0])

            else:
                tmp_array2 += line
        # if "min_distance" in line:
        #     #tmp_array = line.split("number_of_pixels: tensor(")
        #     tmp_array = line.split("min_distance: ")
        #     tmp_array2 = tmp_array[1].split(",")
        #     tmp_array2 = tmp_array2[0].split(".")
        #     print(tmp_array2[0])
        if "result:" in line:
            #tmp_array = line.split("number_of_pixels: tensor(")
            tmp_array = line.split("result: tensor([")
            tmp_array2 = tmp_array[1]
            in_result = 1
    tmp_array3 = np.array(tmp_array3)
    return tmp_array3/tmp_array3.max()


def get_data():
        f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\A-Stern Test\\A-Stern Test\\Assets\\Resources\\single_example_data2.txt", "r")
        # Using readlines()
        Lines = f.readlines()

        weightmatrix = []
        points = []
        target_data_return = []

        count = 0
        # Strips the newline character
        for line in Lines:
            #print("{}".format(line.strip()))
            #print(line)
            elements = line.split(";")

            #get the weightmatrix
            x = elements[0].split(", ")
            x.remove("")
            for i in range(0, len(x), 1):
                x[i] = int(x[i])
            #tensor = torch.tensor(x)
            #weightmatrix = torch.stack((weightmatrix, tensor.view(-1, 10)), dim=0)
            weightmatrix.append(x)

            #get the start and endpoints
            y = elements[1].split(" ")
            for i in range(0, len(y), 1):
                y[i] = y[i].strip('[]')
                y[i] = y[i].split(",")
                for j in range(0, len(y[i]), 1):
                    y[i][j] = int(y[i][j])

            points.append(torch.tensor(y))

            #get the path and transform it into a matrix
            z = elements[2].split(", ")
            z.remove("")

            for i in range(0, len(z), 1):
                z[i] = z[i].strip('()')
                z[i] = z[i].split(",")
                for j in range(0, len(z[i]), 1):
                    z[i][j] = int(z[i][j])

            target_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]              #torch.zeros([10, 10])
            for i in range(0, len(z), 1):
                target_data[z[i][0]][z[i][1]] = 1
            target_data[y[1][0]][y[1][1]] = 1
            target_data_return.append(target_data)

            count += 1
        f.close()

        return weightmatrix, points, target_data_return, count



weightmatrix, points, target_data, batch_size = get_data()              #points = [0,3],[5,7]
weightmatrix = torch.tensor(weightmatrix, dtype=torch.float32)
weightmatrix = weightmatrix.view(batch_size, 1,10,10)



#erste auslesung der Daten
#array1 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight0,5", batch_size=batch_size, points=points, weightmatrix=weightmatrix)
array1 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight0,5", batch_size=batch_size, points=points, weightmatrix=weightmatrix)

#zweite auslesung der Daten
#array2 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight1,1", batch_size=batch_size, points=points, weightmatrix=weightmatrix)
array2 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight1,1", batch_size=batch_size, points=points, weightmatrix=weightmatrix)

#dritte auslesung der Daten
#array3 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight2", batch_size=batch_size, points=points, weightmatrix=weightmatrix)
array3 = determine_loss(path="complete_statistics_changing_gapweight_clusterweight2", batch_size=batch_size, points=points, weightmatrix=weightmatrix)


#x = np.arange(0, 10, 0.1)
#arr = np.array([1, 2, 3, 4, 5])


x = np.arange(0, 10, 0.1)
x2 = np.arange(0, 5000, 50)

#changing gapweight stats
#plt.plot(x2, array1, label="Loss with Clusterweight 0.5")
#plt.plot(x2, array2, label="Loss with Clusterweight 1.1")
plt.plot(x2, array3, label="Loss with Clusterweight 2")

#changing clusterweight stats
# plt.plot(x, array1, label="Loss with Gapweight 1000")
# plt.plot(x, array2, label="Loss with Gapweight 3000")
# plt.plot(x, array3, label="Loss with Gapweight 5000")


plt.xlabel("Weight of the gap-loss")
           # family='serif',
           # color='r',
           # weight='normal',
           # size = 16,
           # labelpad = 6)
plt.ylabel("Normalized Loss")
plt.legend(loc="upper left")
plt.grid()
plt.ylim(-0.1, 1.3)
plt.axhline(color='black', lw=0.75)
plt.axvline(color='black', lw=0.75)
plt.savefig("normalized_loss_changing_gapweight_clusterweight2")
plt.show()
