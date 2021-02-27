import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()

        #the way down the u-form:

        #first convolution-block
        self.down1 = nn.Sequential(
            nn.Conv2d(2, 64, 3),
            nn.Conv2d(64, 64, 3),
            #F.relu()
            #F.max_pool2d(kernel_size=2)
        )

        # self.conv1 = nn.Conv2d(2, 64, 3)
        # self.conv1_1 = nn.Conv2d(64, 64, 3)
            #max pool2d in forward, also safe the state in the forward function

        #second convolution-block
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            #F.relu()
            #F.max_pool2d(kernel_size=2)
        )
        # self.conv2 = nn.Conv2d(64, 128, 3)
        # self.conv2_1 = nn.Conv2d(128, 128, 3)
            #max pool2d in forward, also safe the state in the forward function

        #third convolution-block
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            #F.relu()
            #F.max_pool2d(kernel_size=2)
        )
        # self.conv3 = nn.Conv2d(128, 256, 3)
        # self.conv3_1 = nn.Conv2d(256, 256, 3)
            #max pool2d in forward, also safe the state in the forward function

        #forth convolution-block
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 512, 3),
            #F.relu()
            #F.max_pool2d(kernel_size=2)
        )
        # self.conv4 = nn.Conv2d(256, 512, 3)
        # self.conv4_1 = nn.Conv2d(512, 512, 3)
            #max pool2d in forward, also safe the state in the forward function

        #fifth convolution-block and the bottom of the u-form
        self.bottom = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.Conv2d(1024, 1024, 3),
            #F.relu()
        )
        # self.conv5 = nn.Conv2d(512, 1024, 3)
        # self.conv5_1 = nn.Conv2d(1024, 1024, 3)


        #the way up the u-form, with the help of upconvolution. Use "ConvTranspose2d" and not(!) "Upsample". Upsample doesn't learn, ConvTranspose2d learns parameters

        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.Conv2d(512, 512, 3),
            #F.relu()
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.Conv2d(512, 256, 3),
            nn.Conv2d(256, 256, 3),
            #F.relu()
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.Conv2d(256, 128, 3),
            nn.Conv2d(128, 128, 3),
            #F.relu()
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(128, 64, 3),
            nn.Conv2d(64, 64, 3),
            #F.relu()
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    # x represents our data
    def forward(self, x):
        #down the u
        d_1 = self.down1(x)
        d_1_tmp = d_1
        d_1 = F.max_pool2d(d_1, kernel_size=2)
        d_1 = F.relu(d_1)
        d_2 = self.down2(d_1)
        d_2_tmp = d_2
        d_2 = F.max_pool2d(d_2, kernel_size=2)
        d_2 = F.relu(d_2)
        d_3 = self.down3(d_2)
        d_3_tmp = d_3
        d_3 = F.max_pool2d(d_3, kernel_size=2)
        d_3 = F.relu(d_3)
        d_4 = self.down4(d_3)
        d_4_tmp = d_4
        d_4 = F.max_pool2d(d_4, kernel_size=2)
        d_4 = F.relu(d_4)

        #bottom
        return_value = self.bottom(d_4)
        return_value = F.relu(return_value)

        #up the u
        return_value = self.upconv1(return_value)
        return_value = self.up1(self.concatenate_tensors(return_value, self.crop_tensor(d_4_tmp, 4)))
        return_value = F.relu(return_value)
        return_value = self.upconv2(return_value)
        return_value = self.up2(self.concatenate_tensors(return_value, self.crop_tensor(d_3_tmp, 16)))
        return_value = F.relu(return_value)
        return_value = self.upconv3(return_value)
        return_value = self.up3(self.concatenate_tensors(return_value, self.crop_tensor(d_2_tmp, 40)))
        return_value = F.relu(return_value)
        return_value = self.upconv4(return_value)
        return_value = self.up4(self.concatenate_tensors(return_value, self.crop_tensor(d_1_tmp, 88)))

        return_value = self.final_conv(return_value)

        return return_value

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def crop_tensor(self, x, cropping_size):
        if((x.size()[2]/2) >= cropping_size):
            return x[0:, 0:, cropping_size:-cropping_size, cropping_size:-cropping_size]

    def concatenate_tensors(self, x, y):
        return torch.cat((x, y), 1)                 #dimension 1 is the channel-dimension

def init():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device



if __name__ == '__main__':
    device = init()


    weightmatrix = torch.rand(1,2,572,572)

    target_data = torch.rand(1,1,388,388)

    learning_rate = 0.01

    model_path = 'D:\\Studium\\Bachelorarbeit\\Machine Learning\\ressources\\models'

    nn_pathfinder = U_Net()
    opt = optim.SGD(params=nn_pathfinder.parameters(), lr=learning_rate)
    nn_pathfinder.to(device)

    result = nn_pathfinder(weightmatrix)
    criterion = nn.MSELoss()
    #print ("Result: ", result)

    print("before training")
    # print ("Result: ", result.flatten())
    # print ("Target Data: ",  target_data.flatten())
    print("loss: ", criterion(result, target_data))

    for epoch in tqdm(range(10)):
        result = nn_pathfinder(weightmatrix)

        loss = criterion(result, target_data)
        #if(epoch % 10 == 0):
        #    print("loss" , loss)

        nn_pathfinder.zero_grad()
        loss.backward()
        opt.step()


    print("after")
    # print ("Result: ", result.flatten())
    # #result = result.type(torch.int8)
    # result = torch.round(result)
    # result = result.type(torch.int8)
    # target_data = target_data.type(torch.int8)
    # print ("Result: ", result.view(10,10))
    # print ("Target Data: ",  target_data.view(10,10))
    print("loss: ", loss)

    # save the model
    torch.save(nn_pathfinder.state_dict(),
                model_path + "\\128x128_u-net_nr_" + "1" + ".pth")
