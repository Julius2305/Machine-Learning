import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 100)

    # x represents our data
    def forward(self, x):
        # Flatten x with start_dim=1
        x = x.view(-1, self.num_flat_features(x))
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_data():
        f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\Unity-Projekte\\Generate Data\\Assets\\Resources\\200_training_data.txt", "r") #Insert the path to the data here
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
                points[y[i][0]][y[i][1]] = 1000
            points_return.append(points)

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

        return weightmatrix, points_return, target_data_return, count


weightmatrix, points, target_data, batch_size = get_data()
weightmatrix = torch.tensor(weightmatrix)
weightmatrix = weightmatrix.view(batch_size, 1, 1, 100)

#mark start and endpoint in the weightmatrix
points = torch.tensor(points, dtype=torch.float32)
points = points.view(batch_size, 1,10,10)
weightmatrix = weightmatrix + points.view(batch_size, 1, 1, 100)

learning_rate = 0.1



target_data = torch.tensor(target_data, dtype=torch.float32)
target_data = target_data.view(batch_size, 1, 1, 100)


my_nn = Net()
opt = optim.SGD(params=my_nn.parameters(), lr=learning_rate)

result = my_nn(weightmatrix)
criterion = nn.MSELoss()

print("before training")
print ("Result: ", result.flatten())
print ("Target Data: ",  target_data.flatten())
print("loss: ", criterion(result, target_data))

for epoch in tqdm(range(7500)):
    result = my_nn(weightmatrix)

    loss = criterion(result, target_data)

    my_nn.zero_grad()
    loss.backward()
    opt.step()


print("after")
print ("Result: ", result.flatten())
print ("Target Data: ",  target_data.flatten())
print("loss: ", loss)
