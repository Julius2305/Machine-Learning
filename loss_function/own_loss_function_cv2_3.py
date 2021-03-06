import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

#In this code, the loss of each part of the loss function was summed up. The loss function can work with a variable batch size.

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

            points = points_given[i]

            manhattan_distance_start_end = self.estimate_manhattan_distance(points[0][0], points[0][1], points[1][0], points[1][1])

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
                #print(dist_img[points[0][0]][points[0][1]])

                #get the min distance to the end_cluster from the start_cluster
                min_distance = dist_img[points[0][0]][points[0][1]]
                for cell in start_cluster:
                    if dist_img[cell[0]][cell[1]] < min_distance:
                        min_distance = dist_img[cell[0]][cell[1]]

                gap_loss = min_distance * soa_cells_inv * gap_weight
            else:
                gap_loss = (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight
                gap_loss = gap_loss.view(1)

            cluster_size_penalty = torch.tensor([0], dtype=torch.float32, requires_grad = True)
            cluster_size_penalty = sum((result*weightmatrix).view(100)) * weight_weight * abs(manhattan_distance_start_end - len(start_cluster))

            #Concatenate the loss with the other losses from the batch
            loss = torch.cat((loss, loss_start + gap_loss + cluster_size_penalty), 0)

        #return the mean of all losses over the batch
        return sum(loss)/result_size[0]


    def estimate_manhattan_distance(self, start_x, start_y, end_x, end_y):
        return abs(end_x - start_x) + abs(end_y - start_y)




def get_data():
        f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\Unity-Projekte\\Generate Data\\Assets\\Resources\\200_training_data.txt", "r") #Insert the path to the data here
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

#mark start and end in the weightmatrix
for i in range(0, batch_size):
    weightmatrix[i, 0, points[i][0][0], points[i][0][1]] += 1000
    weightmatrix[i, 0, points[i][1][0], points[i][1][1]] += 1000



random_data = autograd.Variable(torch.rand(1, 1, 10, 10),requires_grad = True)
model_path = 'D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models'

criterion = CustomLoss()

my_nn = Net()
# comment this lines in to evaluate the model
#my_nn.load_state_dict(torch.load("D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models\\own_loss_model_nr_2.pth"))
#my_nn.eval()


#an optimizer can be chosen
# opt = optim.SGD(params=my_nn.parameters(), lr=0.01)
opt = optim.Adam(params=my_nn.parameters(), lr=0.01)
# opt = optim.AdamW(params=my_nn.parameters(), lr=0.01)
# opt = optim.ASGD(params=my_nn.parameters(), lr=0.01)

result = my_nn(weightmatrix.view(batch_size,1, 1,100))

for epoch in tqdm(range(15000)):
    result = my_nn(weightmatrix.view(batch_size,1,1, 100))
    loss = criterion(result_given=result.view(batch_size,1,10,10), points_given=points, weightmatrix_given=weightmatrix)


    my_nn.zero_grad()

    loss.backward()
    #Comment the following lines in to see the gradients of the neural net. The gradients of the last layer(4) can be used as an approximation for the change of the result of the NN.
    # if(sum(my_nn.fc1.bias.grad) != 0):
    #     print(my_nn.fc1.bias.grad)
    # if(sum(my_nn.fc2.bias.grad) != 0):
    #     print(my_nn.fc2.bias.grad)
    # if(sum(my_nn.fc3.bias.grad) != 0):
    #     print(my_nn.fc3.bias.grad)
    # if(sum(my_nn.fc4.bias.grad) != 0):
    #     #pass
    #     print(my_nn.fc4.weight.grad)
    #     weights = my_nn.fc4.bias.grad *1000
        # #else:
        #     #print(len(weights))
        #     #weights_img = weights.view(250,100)
        #     weights_img = weights.view(10,10)
        #     weights_imgplot = plt.matshow(weights_img.detach().numpy())
        #     plt.show()

    #save the model
    opt.step()
    torch.save(my_nn.state_dict(),
                model_path + "\\own_loss_model_nr_" + "4" + ".pth")

# print("loss" , loss)
# print(result.view(10,10))
# #print(torch.round(result).view(10,10))
# img = result.round().view(10,10)
# imgplot = plt.matshow(img.detach().numpy())
# plt.show()
