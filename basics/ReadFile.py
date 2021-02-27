import torch

if __name__=="__main__":
    f = open("D:\\Studium\\Bachelorarbeit\\Unity Projekte\\A-Stern Test\\A-Stern Test\\Assets\\Resources\\data.txt", "r")
    # Using readlines()
    Lines = f.readlines()

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
        tensor = torch.tensor(x)
        weightmatrix = tensor.view(-1, 10)

        #get the start and endpoints
        y = elements[1].split(" ")
        for i in range(0, len(y), 1):
            y[i] = y[i].strip('[]')
            y[i] = y[i].split(",")
            for j in range(0, len(y[i]), 1):
                y[i][j] = int(y[i][j])

        points = torch.tensor(y)

        #get the path and transform it into a matrix
        z = elements[2].split(", ")
        z.remove("")
        print(z)

        for i in range(0, len(z), 1):
            z[i] = z[i].strip('()')
            z[i] = z[i].split(",")
            for j in range(0, len(z[i]), 1):
                z[i][j] = int(z[i][j])

        target_data = torch.zeros([10, 10])
        for i in range(0, len(z), 1):
            target_data[z[i][0]][z[i][1]] = 1
        target_data[y[1][0]][y[1][1]] = 1

        print(target_data)
