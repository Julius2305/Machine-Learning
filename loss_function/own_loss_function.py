import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm

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
        #x = self.dropout2(x)
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

    def forward(self, result_given, points_given):
        #variables for easier variation of the loss function
        weight = 1000
        lonelyness_weight = 15
        cluster_weight = 5      #should be a smaller weight than lonelyness_weight (denn die Strafe dafür in einem Cluster nicht eins zu sein sollte kleiner sein, als die Strafe dafür in gar keinem Cluster einen Wert >0 zu haben)
        cell_weight = 20        #must be greater than lonelyness_weight (dadurch wird es weniger dafür bestraft eine eins zu sein, als keine eins in dem rechteck zwischen den Start/End Punkten zu sein)
        cluster_size_weight = 12 # muss kleiner als lonelyness_weight sein!!! Wirkt invers zu cluster_weight. Very sensible. Idea: Could be a higher penalty depending on the distance between start and endpoint
        gap_weight = 300
        result_size = result_given.size()
        loss = torch.tensor([0], dtype=torch.float32)

        for i in range(0, result_size[0]):
            result = result_given[i, 0, 0:, 0:]
            points = points_given[i]
            if i%100 == 0:
                print(i)

            manhattan_distance_start_end_y = points[1][0] - points[0][0]
            manhattan_distance_start_end_x = points[1][1] - points[0][1]
            manhattan_distance_start_end = self.estimate_manhattan_distance(points[0][0], points[0][1], points[1][0], points[1][1])

            #set the start and endpoint to 1
            loss_start = (2-(result[points[0][0]][points[0][1]] + result[points[1][0]][points[1][1]])) * weight

            #first compute the clusters
            start_cluster = [[points[0][0], points[0][1]]]  #here: first y-value, 2nd x-value

            for cell in start_cluster:
                #add neighborcells into the start_cluster
                #left side of the current cell
                if cell[1] != 0:
                    if result[cell[0]][cell[1]-1].round() == 1:
                        if [cell[0], cell[1]-1] not in start_cluster:
                            start_cluster.append([cell[0], cell[1]-1])
                    if cell[0] != 0:
                        if result[cell[0] - 1][cell[1] - 1].round() == 1:
                            if [cell[0] - 1, cell[1] - 1] not in start_cluster:
                                start_cluster.append([cell[0] - 1, cell[1] -1])
                    if cell[0] != result.size()[0] - 1:
                        if result[cell[0] + 1][cell[1] - 1].round() == 1:
                            if [cell[0] + 1, cell[1] - 1] not in start_cluster:
                                start_cluster.append([cell[0] + 1, cell[1] -1])

                #right side of the current cell
                if cell[1] != result.size()[1] - 1:
                    if result[cell[0]][cell[1] + 1].round() == 1:
                        if [cell[0], cell[1] + 1] not in start_cluster:
                            start_cluster.append([cell[0], cell[1] + 1])
                    if cell[0] != 0:
                        if result[cell[0] - 1][cell[1] + 1].round() == 1:
                            if [cell[0] - 1, cell[1] + 1] not in start_cluster:
                                start_cluster.append([cell[0] - 1, cell[1] + 1])
                    if cell[0] != result.size()[0] - 1:
                        if result[cell[0] + 1][cell[1] + 1].round() == 1:
                            if [cell[0] + 1, cell[1] + 1] not in start_cluster:
                                start_cluster.append([cell[0] + 1, cell[1] + 1])

                #upper side of the cell
                if cell[0] != 0:
                    if result[cell[0] - 1][cell[1]].round() == 1:
                        if [cell[0] - 1, cell[1]] not in start_cluster:
                            start_cluster.append([cell[0] - 1, cell[1]])

                #down side of the cell
                if cell[0] != result.size()[0] - 1:
                    if result[cell[0] + 1][cell[1]].round() == 1:
                        if [cell[0] + 1, cell[1]] not in start_cluster:
                            start_cluster.append([cell[0] + 1, cell[1]])

            #the start cluster is now created



            #use the penalty to form a square between start and endpoint as first approximation of the path. The points outside increase the loss if they are greater 1. This loss increases further away from the path/square.
            single_cell_penalty = torch.tensor([0], dtype=torch.float32)
            row_counter = -1
            column_counter = -1
            for row in result:
                row_counter += 1
                for column in row:
                    column_counter += 1
                    m_distance_to_start_and_end = self.estimate_manhattan_distance(row_counter, column_counter, points[0][0], points[0][1]) + self.estimate_manhattan_distance(row_counter, column_counter, points[1][0], points[1][1])
                    #manhattan distance to both points summed is always greater or equal than the manhattan distance between these both points. It's equal in a square between them, its greater if a point is not in this square
                    if (m_distance_to_start_and_end - manhattan_distance_start_end == 0):
                        single_cell_penalty += (1-column) * cell_weight #* (column + ((self.estimate_manhattan_distance(column_counter, row_counter, points[0][0], points[0][1]) + self.estimate_manhattan_distance(column_counter, row_counter, points[1][0], points[1][1])) - manhattan_distance_start_end - 1))
                    else:
                        single_cell_penalty += column * ((m_distance_to_start_and_end - manhattan_distance_start_end)/2)

                column_counter = -1



            #estimate the single_cell_penalty -> penalty for being a 1 without being connected to (element of) the start_cluster.
            #penalty for numbers outside the clusters
            lonelyness_penalty = torch.tensor([0], dtype=torch.float32)
            cluster_size_penalty = torch.tensor([0], dtype=torch.float32)
            row_counter = -1
            column_counter = -1
            for row in result:
                row_counter += 1
                for column in row:
                    column_counter += 1
                    if not ([column_counter, row_counter] in start_cluster):
                        lonelyness_penalty += column * lonelyness_weight
                    else:
                        #penalty, so that the entries in the clusters move to 1
                        lonelyness_penalty += (1-column) * cluster_weight
                        #penalty, so that there are not too many entries in the cluster
                        cluster_size_penalty += column * len(start_cluster) * cluster_size_weight

                    if(column_counter == points[0][0] and row_counter == points[0][1]):
                        pass

                column_counter = -1


            #create a sum of all cell_values to use it as a possible penalty
            soa_cells = torch.tensor([0], dtype=torch.float32)                  #sum of all cells
            soa_cells_inv = torch.tensor([0], dtype=torch.float32)                  #sum of all cells inverted
            for row in result:
                for column in row:
                    soa_cells += column
                    soa_cells_inv += (1-column)


            #Multipliziere diese mit der strafe für die Gap. Berechne diese Strafe mit einem gap_weight und der Entfernung des nächsten elements des start_clusters zum Ziel.
            gap_size = manhattan_distance_start_end
            nearest_cell = [points[0][0], points[0][1]]
            for cell in start_cluster:
                tmp_gap_size = self.estimate_manhattan_distance(cell[0], cell[1], points[1][0], points[1][1])
                if tmp_gap_size < gap_size:
                    gap_size = tmp_gap_size
                    nearest_cell = [cell[0], cell[1]]


            #search for the next cell in the gap. it is a neighbour of the "nearest_cell". You can use its value as a (inverted) penalty to set it to 1. First find the offset to the goal
            offset_y = points[1][0] - nearest_cell[0]
            offset_x = points[1][1] - nearest_cell[1]
            #if the y-offset is positive, the y-value of the next cell must be higher than the y-value of "nearest_cell" (same for x-value)

            next_cell = nearest_cell
            gap_next_cell = gap_size
            #left side of the nearest cell
            if offset_x < 0:
                if (self.estimate_manhattan_distance(nearest_cell[0], nearest_cell[1]-1, points[1][0], points[1][1]) < gap_next_cell):
                    next_cell = [nearest_cell[0], nearest_cell[1]-1]
                    gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0], nearest_cell[1]-1, points[1][0], points[1][1])
                if nearest_cell[0] != 0:
                    if (self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1]-1, points[1][0], points[1][1]) < gap_next_cell):
                        next_cell = [nearest_cell[0]-1, nearest_cell[1]-1]
                        gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1]-1, points[1][0], points[1][1])
                if cell[0] != result.size()[0] - 1:
                    if (self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1]-1, points[1][0], points[1][1]) < gap_next_cell):
                        next_cell = [nearest_cell[0]+1, nearest_cell[1]-1]
                        gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1]-1, points[1][0], points[1][1])

            #right side of the nearest cell
            if offset_x > 0:
                if (self.estimate_manhattan_distance(nearest_cell[0], nearest_cell[1]+1, points[1][0], points[1][1]) < gap_next_cell):
                    next_cell = [nearest_cell[0], nearest_cell[1]+1]
                    gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0], nearest_cell[1]+1, points[1][0], points[1][1])
                if nearest_cell[0] != 0:
                    if (self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1]+1, points[1][0], points[1][1]) < gap_next_cell):
                        next_cell = [nearest_cell[0]-1, nearest_cell[1]+1]
                        gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1]+1, points[1][0], points[1][1])
                if cell[0] != result.size()[0] - 1:
                    if (self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1]+1, points[1][0], points[1][1]) < gap_next_cell):
                        next_cell = [nearest_cell[0]+1, nearest_cell[1]+1]
                        gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1]+1, points[1][0], points[1][1])

            #upper side of the cell
            if offset_y < 0:
                if (self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1], points[1][0], points[1][1]) < gap_next_cell):
                    next_cell = [nearest_cell[0]-1, nearest_cell[1]]
                    gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]-1, nearest_cell[1], points[1][0], points[1][1])

            #down side of the cell
            if offset_y > 0:
                if (self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1], points[1][0], points[1][1]) < gap_next_cell):
                    next_cell = [nearest_cell[0]+1, nearest_cell[1]]
                    gap_next_cell = self.estimate_manhattan_distance(nearest_cell[0]+1, nearest_cell[1], points[1][0], points[1][1])


            gap_penalty = gap_size * gap_weight * (1-result[next_cell[0]][next_cell[1]])

            #TODO-Idea: Eine Strafe für die Nähe zu den beiden nicht start/end - ecken des erzeugten vierecks -> Soll den Pfad dünner machen und mehr in die Mitte drücken.


            loss = loss_start + lonelyness_penalty + single_cell_penalty + cluster_size_penalty + gap_penalty + loss

        return loss


    def estimate_manhattan_distance(self, start_x, start_y, end_x, end_y):
        return abs(end_x - start_x) + abs(end_y - start_y)




def get_data():
        f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\A-Stern Test\\A-Stern Test\\Assets\\Resources\\single_example_data.txt", "r")
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

random_data = autograd.Variable(torch.rand(1, 1, 10, 10))
model_path = 'D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models'

criterion = CustomLoss()

my_nn = Net()
my_nn.load_state_dict(torch.load("D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models\\own_loss_model_nr_1.pth"))
my_nn.eval()



opt = optim.SGD(params=my_nn.parameters(), lr=0.1)
result = my_nn(weightmatrix.view(batch_size,1, 1,100))

for epoch in tqdm(range(15000)):
    result = my_nn(weightmatrix.view(batch_size,1,1, 100))
    loss = criterion(result_given=result.view(batch_size,1,10,10), points_given=points)
    print("loss" , loss)

    my_nn.zero_grad()
    loss.backward()
    opt.step()
    torch.save(my_nn.state_dict(),
                model_path + "\\own_loss_model_nr_" + "1" + ".pth")

print(result.view(10,10))
print(torch.round(result).view(10,10))
