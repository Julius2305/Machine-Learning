import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 100)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        #x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = x.view(-1, self.num_flat_features(x))

        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def get_data():
        f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\A-Stern Test\\A-Stern Test\\Assets\\Resources\\single_example_data.txt", "r")
        # Using readlines()
        Lines = f.readlines()

        weightmatrix = []
        points_return = []
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
            points = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

            y = elements[1].split(" ")
            for i in range(0, len(y), 1):
                y[i] = y[i].strip('[]')
                y[i] = y[i].split(",")
                for j in range(0, len(y[i]), 1):
                    y[i][j] = int(y[i][j])
                points[y[i][0]][y[i][1]] = 1
            points_return.append(points)

            #points.append(torch.tensor(y))

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
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            for i in range(0, len(z), 1):
                target_data[z[i][0]][z[i][1]] = 1
            target_data[y[1][0]][y[1][1]] = 1
            target_data_return.append(target_data)

            count += 1
        f.close()

        return weightmatrix, points_return, target_data_return, count


if __name__=="__main__":
    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results
    model = Net()
    model.load_state_dict(torch.load("D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models\\model_nr_2.pth"))
    model.eval()

    weightmatrix, points, target_data, batch_size = get_data()
    weightmatrix = torch.tensor(weightmatrix, dtype=torch.float32)
    weightmatrix = weightmatrix.view(batch_size, 1,10,10)

    points = torch.tensor(points, dtype=torch.float32)
    points = points.view(batch_size, 1,10,10)

    #merge the weightmatrix and the start-/endpoints to a two-channeled input
    weightmatrix = torch.cat((weightmatrix, points), 1)

    target_data = torch.tensor(target_data, dtype=torch.float32)
    target_data = target_data.view(batch_size, 100)

    result = model(weightmatrix)

    criterion = nn.MSELoss()

    print("loss: ", criterion(result, target_data))
    #print ("Result: ", result.view(10,10))
    #result = result * 7
    result = torch.round(result)
    #print(np.reshape(result.detach().numpy(), (10,10)))
    result_plot = plt.matshow(np.reshape(result.detach().numpy(), (10,10)), cmap=plt.cm.gray)

    result = result.type(torch.int8)

    target_data_plot = plt.matshow(np.reshape(target_data.detach().numpy(), (10,10)), cmap=plt.cm.gray)
    plt.show()

    target_data = target_data.type(torch.int8)
    print ("Result: ", result.view(10,10))
    print ("Target Data: ",  target_data.view(10,10))
