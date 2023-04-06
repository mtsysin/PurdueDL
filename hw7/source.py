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
from typing import Any, Callable, List, Optional, Type, Union
from operator import add
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import time
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


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
num_epochs = 50
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
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
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
            nn.Conv2d(64 * 8, 1, 2, 0, 0, bias=False),
        )
        self.final = nn.Linear(batch_size * 4, 1)
    def forward(self, x):              
        x = self.main(x)
        print(x.size())
        x = x.view(-1)
        x = self.final(x)
        x = x.mean(0)       
        x = x.view(1)
        return x
    
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
    
def show_imgs():
    # Plot some training images
    real_batch = next(iter(dataloader))
    print(real_batch[0].size())
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    imgs_save = np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True, nrow=4).cpu(),(1,2,0)).numpy()
    plt.imsave("out/train_imgs.png", imgs_save)


def train_bce():

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

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
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

            # (2) Update G network: maximize log(D(G(z)))
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
            if (iters % 250 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
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
    plt.savefig("out/figure.png")


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ims = [[plt.imsave(f"out/imgs/{index}.png" , np.transpose(i,(1,2,0)).numpy())] for index, i in enumerate(img_list)]

    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())


def wgan():
    """
    This function is meant for training a CG1-based Critic-Generator WGAN.   The implementation
    shown uses several programming constructs from the WGAN implementation at GitHub by the
    original authors of the famous WGAN paper. I have also used several programming constructs 
    from the DCGAN code at PyTorch and GitHub.  Regarding how to set the parameters of this method, 
    see the following script in the "ExamplesAdversarialLearning" directory of the distribution:

                    wgan_CG1.py
    """


    # Create the generator
    netG = Generator().to(device)
    # Create the Discriminator
    netC = Critic().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netC.apply(weights_init)
    netG.apply(weights_init)

    # Print the model
    print(netG)
    print(netC)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerC = torch.optim.Adam(netC.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    

    #  These are for training the Critic, 'one' is for the part of the training with actual
    #  training images, and 'minus_one' is for the part based on the images produced by the 
    #  Generator:
    one = torch.FloatTensor([1]).to(device)
    minus_one = torch.FloatTensor([-1]).to(device)
    # one = torch.tensor(1, dtype=torch.float)
    # minus_one = torch.tensor(-1, dtype=torch.float)


    img_list = []                               
    Gen_losses = []                               
    Cri_losses = []                               
    iters = 0                                   
    gen_iterations = 0
    print("\n\nStarting Training Loop.......[Be very patient at the beginning since the Critic must separately be taken through a few hundred iterations of training before you get to see anything displayed in your terminal window.  Depending on your hardware, it may take around 5 minutes. Subsequently, each 100 iterations will take just a few seconds. ]\n\n")      
    start_time = time.perf_counter()            
    clipping_thresh = 0.01
    # For each epoch
    for epoch in range(num_epochs):        
        data_iter = iter(dataloader)
        i = 0
        ncritic = 5
        while i < len(dataloader):
            for p in netC.parameters():
                p.requires_grad = True          
            if gen_iterations < 25 or gen_iterations % 500 == 0:    # the choices 25 and 500 are from WGAN
                ncritic = 100
            ic = 0
            ##  The inner 'while' loop shown below calculates the expectations in Eq. (8) in the doc section
            ##  at the beginning of this file:
            while ic < ncritic and i < len(dataloader):
                ic += 1
                for p in netC.parameters():
                    p.data.clamp_(-clipping_thresh, clipping_thresh)
                ## Training the Critic (Part 1):
                #  The maximization needed for training the Critic, as shown in Eq. (8) in the doc section
                #  at the beginning of this file, consists of two parts.  The first part involves applying the
                #  Critic network to just the training images, with each image subject to a "gradient
                #  target" of "-1".
                netC.zero_grad()                                                                            
                real_images_in_batch =  next(data_iter)
                i += 1
                real_images_in_batch =  real_images_in_batch[0].to(device)   
                #  Need to know how many images we pulled in since at the tailend of the dataset, the 
                #  number of images may not equal the user-specified batch size:
                b_size = real_images_in_batch.size(0)   
                #  Note that a single scalar is produced for all the data in a batch.  This is probably
                #  the reason why what the Generator learns is somewhat fuzzy.
                critic_for_reals_mean = netC(real_images_in_batch)
                ## 'minus_one' is the gradient target:
                critic_for_reals_mean.backward(minus_one)  

                ## Training the Critic (Part 2):
                #  The second part of Critic training requires that we apply the Critic to the images
                #  produced by the Generator for a fresh batch of input noise vectors. The output of 
                #  the Critic for these images must be subject to the target "-1".
                noise = torch.randn(b_size, nz, 1, 1, device=device)    
                fakes = netG(noise)          
                #  Again, a single number is produced for the whole batch:
                critic_for_fakes_mean = netC(fakes)
                ## 'one' is the gradient target:
                critic_for_fakes_mean.backward(one)
                wasser_dist = critic_for_reals_mean - critic_for_fakes_mean
                loss_critic = critic_for_fakes_mean - critic_for_reals_mean
                #  Update the Critic
                optimizerC.step()   

            ## Training the Generator:
            ##   That brings us to the training of the Generator through the minimization required by the 
            ##   minmax objective in Eq. (7) at the beginning of this file.  To that end, first we must 
            ##   turn off the "requires_grad" of the Critic parameters since the Critic and the Generator 
            ##   must be updated independently:
            for p in netC.parameters():
                p.requires_grad = False
            netG.zero_grad()                         
            #  This is again a single scalar based characterization of the whole batch of the Generator images:
            noise = torch.randn(b_size, nz, 1, 1, device=device)    
            fakes = netG(noise)          
            critic_for_fakes_mean = netC(fakes)
            loss_gen = critic_for_fakes_mean
            critic_for_fakes_mean.backward(minus_one)                       
            #  Update the Generator
            optimizerG.step()                                                                          
            gen_iterations += 1

            if i % (ncritic * 20) == 0:   
                current_time = time.perf_counter()                                                            
                elapsed_time = current_time - start_time                                                      
                print("[epoch=%d/%d   i=%4d   el_time=%5d secs]     loss_critic=%7.4f   loss_gen=%7.4f   Wasserstein_dist=%7.4f" %  (epoch, num_epochs,i,elapsed_time,loss_critic.data[0], loss_gen.data[0], wasser_dist.data[0]))
            Gen_losses.append(loss_gen.data[0].item())      
            Cri_losses.append(loss_critic.data[0].item())   
            #  Get G's output on fixed_noise for the GIF animation:
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)): 
                with torch.no_grad():                                                                        
                    fake = netG(fixed_noise).detach().cpu()  ## detach() removes the fake from comp. graph.
                                                                ## for its CPU compatible version
                img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))   
            iters += 1                                                                                        
    
    #  At the end of training, make plots from the data in Gen_losses and Cri_losses:
    plt.figure(figsize=(10,5))                                                                             
    plt.title("Generator and Critic Loss During Training")                                          
    plt.plot(Gen_losses,label="G")                                                                           
    plt.plot(Cri_losses,label="C")                                                                           
    plt.xlabel("iterations")                                                                               
    plt.ylabel("Loss")                                                                                     
    plt.legend()                                                                                           
    plt.savefig("out/figure_wgan.png")                                  
    plt.show()                                                                                             
    #  Make an animated gif from the Generator output images stored in img_list:  
    #
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ims = [[plt.imsave(f"out/imgs/{index}.png" , np.transpose(i,(1,2,0)).numpy())] for index, i in enumerate(img_list)]  


    # images = []                                                                                            
    # for imgobj in img_list:                                                                                
    #     img = tvtF.to_pil_image(imgobj)                                                                    
    #     images.append(img)                                                                                 
    # imageio.mimsave(dir_name_for_results + "/generation_animation.gif", images, fps=5)                     
    
    # #  Make a side-by-side comparison of a batch-size sampling of real images drawn from the
    # #  training data and what the Generator is capable of producing at the end of training:
    # real_batch = next(iter(dataloader))                                                        
    # real_batch = real_batch[0]
    # plt.figure(figsize=(15,15))                                                                           
    # plt.subplot(1,2,1)                                                                                    
    # plt.axis("off")                                                                                       
    # plt.title("Real Images")                                                                              
    # plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch.to(self.device), 
    #                                     padding=1, pad_value=1, normalize=True).cpu(),(1,2,0)))  
    # plt.subplot(1,2,2)                                                                             
    # plt.axis("off")                                                                                
    # plt.title("Fake Images")                                                                       
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))                                                 
    # plt.savefig(dir_name_for_results + "/real_vs_fake_images.png")                                 
    # plt.show()                                                                                     


def get_fid(real_paths, fake_paths):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims] 
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics( real_paths, model, device=device)
    m2, s2 = calculate_activation_statistics( fake_paths, model, device=device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print ('FID: {:.2f}'.format(fid_value))

def generate_1000(model):
    im_path = os.path.join(DATA_ROOT, "train")
    im
    for i in range(1000):
        fixed_noise = torch.randn(32, 100, 1, 1, device=device)


if __name__=="__main__":
    train_bce()