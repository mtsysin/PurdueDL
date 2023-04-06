# from ComputationalGraphPrimer import *
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as tvt
import tqdm
from sklearn.metrics import confusion_matrix
import cv2
import math
from dataset import COCODataset
from loss import YOLOLoss
from typing import Any, Callable, List, Optional, Type, Union
from model import ResnetBlock, HW5Net
from operator import add
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision


MIN_W = 200
MIN_H = 200
DATA_ROOT = "./pizzas"
LOSS_COUNT = 50
image_size = 64

# Number of workers for dataloader
workers = 1
# Batch size during training
batch_size = 128
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of training epochs
num_epochs = 5
# Length of noise vector
nz = 100

# Initialize weights:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Networks (Generator, Discriminator)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


if __name__=="__main__":

    # Set up the dataset for training
    dataset = dset.ImageFolder(root=DATA_ROOT,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    print(real_batch[0].size())
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    imgs_save = np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True, nrow=4).cpu(),(1,2,0)).numpy()
    plt.imsave("out/train_imgs.png", imgs_save)



    # Create the generator
    netG = Generator().to(device)
    # Create the Discriminator
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    netG.apply(weights_init)

    # Print the model
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    img_list = []                               
    G_losses = []                               
    D_losses = []                               
    iters = 0   

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netD.state_dict(), 'models/netD')
    torch.save(netG.state_dict(), 'models/netG')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# def train(net, save = False):
#     # Choose device
#     if torch.cuda.is_available()== True: 
#         device = torch.device("cuda:0")
#     else: 
#         device = torch.device("cpu")

#     net.train()

#     # Create transform
#     transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     batch = 10

#     train_dataset = COCODataset(
#         root=ROOT,
#         categories_list=['bus', 'cat', 'pizza'],
#         download = False,
#         verify = True,
#         train = True,
#         transform=transform,
#     )

#     train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
#                                                   batch_size = batch, 
#                                                   shuffle = True, 
#                                                   num_workers = 0)

#     net = net.to(device)

#     criterion = YOLOLoss()
#     # criterion_localization = torch.nn.MSELoss(reduction="sum")

#     optimizer = torch.optim.Adam(
#         net.parameters(), 
#         lr=1e-3, 
#         betas=(0.9, 0.99)
#     )

#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    
#     losses = []
#     losses_separate = []

#     epochs = 5

#     file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
#     outer = tqdm.tqdm(total=epochs, desc='Epochs', position=0)
#     for epoch in range(epochs):
#         running_loss = 0.0
#         running_loss_separate = [0.0] * 3
#         inner = tqdm.tqdm(total=len(train_data_loader), desc='Batches', position=0)
#         for i, data in enumerate(train_data_loader):
#             inputs, labels = data
#             inputs = inputs.to(device)

#             # print(inputs[0, 0, 0,...])

#             labels = labels.to(device) 

#             optimizer.zero_grad()

#             outputs = net(inputs)

#             # print(outputs[0, 0, 0,...])

#             batch_losses = criterion(outputs, labels)
#             # print(batch_losses)

#             loss = sum(batch_losses)
#             # print(loss)

#             loss.backward()
#             # for param in net.parameters():
#                 # print(param.grad[0,...], param.size())
#             #     break
#             optimizer.step()

#             running_loss += loss.item()
#             running_loss_separate = [curr_loss + new_loss.item() for curr_loss, new_loss in zip(running_loss_separate, batch_losses)]


#             if (i+1) % LOSS_COUNT == 0:
#                 file_log.set_description_str(
#                     "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / LOSS_COUNT)
#                 )
#                 losses.append(running_loss / LOSS_COUNT)
#                 losses_separate.append([el/LOSS_COUNT for el in running_loss_separate])
#                 running_loss = 0.0
#                 running_loss_separate = [0.0] * 3

#                 # print("Labels bboxes", labels_bboxes)
#                 # print("Labels classes", labels_classes)
#                 # print("OUT bboxes", output_bboxes)
#                 # print("OUT classes", output_classes)

#             inner.update(1)
#         scheduler.step()
#         outer.update(1)

#     if save:
#         torch.save(net.state_dict(), ROOT+'/model')

#     return losses, losses_separate


    
# if __name__=="__main__":
#     # Initialization
#     # seed = 0
#     # random.seed(seed)
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed(seed)
#     # np.random.seed(seed)
#     # os.environ['PYTHONHASHSEED'] =  str(seed)

#     class_list = ['bus', 'cat', 'pizza']

#     net = HW5Net(3)
#     net = net.to(torch.float32)
#     net.load_state_dict(torch.load(ROOT+'/models/model5', map_location=torch.device('cpu')))

#     # loss_trace, loss_traces = train(net, save=True)
#     # labels_names = ["BCE", "MSE", "Cross entropy"]
#     # plt.plot(loss_trace, label = "Combined")
#     # for i, trace in enumerate(zip(*loss_traces)):
#     #     plt.plot(trace, label = labels_names[i])

#     # plt.ylabel('Loss')
#     # plt.xlabel('Processed batches * nz')
#     # plt.legend()
#     # plt.savefig("./out/loss_trace1.png")

#     transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     dataset = COCODataset(
#         root=ROOT,
#         categories_list=class_list,
#         train = False,
#         clear = False,
#         download = False,
#         verify = True,
#         grid_size = 64,
#         return_raw = False,
#         anchor_boxes = 5,
#         transform=transform
#     )
#     raw_dataset = COCODataset(
#         root=ROOT,
#         categories_list=class_list,
#         train = False,
#         clear = False,
#         download = False,
#         verify = True,
#         grid_size = 64,
#         return_raw = False,
#         anchor_boxes = 5,
#     )

#     MAX_SHOW = 3

#     for _ in range(nz):
#         # Get index from the dataset
#         idx =  np.random.randint(0, len(dataset)) # 452, 2343 388 97 136 3284
#         # idx= 136
#         print(idx)
#         image, _ = raw_dataset[idx]
#         image_input , label = dataset[idx]

#         label = net(image_input.unsqueeze(0)).squeeze(0)

#         print("Image size: ", image.size)
#         print("Label size: ", label.size())
#         # print(label[..., 0])

#         # Find indices and bboxes where there is an image:
#         pred_bce = nn.Sigmoid()(label[..., 0])
#         # top = torch.topk(pred_bce, 3, dim=-1)
#         # print("TOP: ", top)
#         Iobj_i = (nn.Sigmoid()(label[..., 0])>0.1).bool()

#         selected_igms = label[Iobj_i]
#         _, select_ind = torch.topk(selected_igms[..., 0], 2)
#         print(select_ind)
#         selected_igms = selected_igms[select_ind]
#         selected_igms_positions = Iobj_i.nonzero(as_tuple=False)
#         selected_igms_positions =selected_igms_positions[select_ind]
#         print("Selected yolo vectors: ", selected_igms)
#         print("Selected yolo vectors positions: ", selected_igms_positions)

#         # Show image and corresponding bounding boxes:
#         image = np.uint8(image)
#         fig, ax = plt.subplots(1, 1)
#         for yolo_vector, position in zip(selected_igms, selected_igms_positions):
#             # get bbox values and convert them to scalars
#             x, y, w, h = yolo_vector[1:5].tolist()
#             class_vector = yolo_vector[5:].tolist()
#             class_index = class_vector.index(max(class_vector))
#             anchor_idx, x_idx, y_idx = position.tolist()

#             if anchor_idx == 0: 
#                 w_scale, h_scale = 3, 1
#             if anchor_idx == 1: 
#                 w_scale, h_scale = 2, 1
#             if anchor_idx == 2: 
#                 w_scale, h_scale = 1, 1
#             if anchor_idx == 3: 
#                 w_scale, h_scale = 1, 2
#             if anchor_idx == 4: 
#                 w_scale, h_scale = 1, 3

#             # Select correct dimenstions
#             x = int((x + (x_idx + 0.5)) * dataset.grid_size)
#             y = int((y + (y_idx + 0.5)) * dataset.grid_size)
#             w = int(math.exp(w) * dataset.grid_size * w_scale)
#             h = int(math.exp(h) * dataset.grid_size * h_scale)

#             print(x, y, w, h)

#             image = cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (36, 255, 12), 2) 
#             image = cv2.putText(image, class_list[class_index], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (36, 255, 12), 2)


#         ax.imshow(image) 
#         ax.set_axis_off() 
#         plt.axis('tight') 
#         plt.show()

