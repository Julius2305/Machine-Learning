#This script tracks the stats of the final neural net with an adopted lossfunction

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 512)
        self.fc3 = nn.Linear(512, 250)
        self.fc4 = nn.Linear(250, 100)

    # x represents our data
    def forward(self, x):
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CustomLoss(nn.Module):
    def init(self):
        super(CustomLoss,self).__init__()

    def forward(self, result_given, points_given, weightmatrix, weight_weight):
        #variables for easier variation of the loss function
        weight = 20000
        gap_weight = 5000 #3000 #1200
        cluster_size_weight = 3
        result_size = result_given.size()

        for i in range(0, result_size[0]):
            result = result_given[i, 0, 0:, 0:]
            points = points_given[i]

            manhattan_distance_start_end = self.estimate_manhattan_distance(points[0][0], points[0][1], points[1][0], points[1][1])
            min_distance = manhattan_distance_start_end

            soa_cells = sum(sum(result))
            soa_cells_inv = 100 - sum(sum(result))

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

            start_value = labels_im[points[0][0]][points[0][1]]
            end_value = labels_im[points[1][0]][points[1][1]]

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


                #get the min distance to the end_cluster from the start_cluster
                min_distance = dist_img[points[0][0]][points[0][1]]
                for cell in start_cluster:
                    if dist_img[cell[0]][cell[1]] < min_distance:
                        min_distance = dist_img[cell[0]][cell[1]]
                #print(min_distance)

                gap_loss = min_distance * soa_cells_inv * gap_weight
            else:
                gap_loss = (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight
                gap_loss = gap_loss.view(1)


            cluster_size_penalty = torch.tensor([0], dtype=torch.float32, requires_grad = True)
            cluster_size_penalty = sum((result*weightmatrix).view(100)) * weight_weight * abs(manhattan_distance_start_end - len(start_cluster))

        return loss_start + cluster_size_penalty + gap_loss , min_distance


    def estimate_manhattan_distance(self, start_x, start_y, end_x, end_y):
        return abs(end_x - start_x) + abs(end_y - start_y)




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
            elements = line.split(";")

            #get the weightmatrix
            x = elements[0].split(", ")
            x.remove("")
            for i in range(0, len(x), 1):
                x[i] = int(x[i])

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



weightmatrix, points, target_data, batch_size = get_data()
weightmatrix = torch.tensor(weightmatrix, dtype=torch.float32)
weightmatrix = weightmatrix.view(batch_size, 1,10,10)



random_data = autograd.Variable(torch.rand(1, 1, 10, 10),requires_grad = True)
model_path = 'D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models'

criterion = CustomLoss()

for i in range(0, 100, 1):
    my_nn = Net()


    #an optimizer can be chosen
    # opt = optim.SGD(params=my_nn.parameters(), lr=0.01)
    opt = optim.Adam(params=my_nn.parameters(), lr=0.01)
    # opt = optim.AdamW(params=my_nn.parameters(), lr=0.01)
    # opt = optim.ASGD(params=my_nn.parameters(), lr=0.01)

    result = my_nn(random_data.view(batch_size,1, 1,100))

    for epoch in tqdm(range(15000)):
        result = my_nn(random_data.view(batch_size,1,1, 100))

        loss, min_distance = criterion(result_given=result.view(batch_size,1,10,10), points_given=points, weightmatrix=weightmatrix, weight_weight=(i/10))

        my_nn.zero_grad()

        loss.backward()

        opt.step()


    print("loss" , loss)
    print(result.view(10,10))
    f = open("complete_statistics_changing_clusterweight_gapweight5000.txt", "a")
    to_write = "clusterweight: " + str(i/10)  + ", min_distance: " + str(min_distance) + ", result: " + str(result.view(100)) + ", result_rounded: " + str(result.round().view(100)) + ", number_of_pixels: " + str(sum(result.round().view(100))) + "\n"
    f.write(to_write)
    f.close()
