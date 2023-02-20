import os
from PIL import Image
import torch
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import torchvision.transforms as tvt
import random
import time
from scipy.stats import wasserstein_distance

ROOT = "./imgs"

class MyDataset (torch.utils.data.Dataset):
    def __init__ (self, root, raw = False):
        # raw is a helper variable
        super () . __init__ ()
        # Obtain meta information (e.g. list of file names)
        # Initialize data augmentation transforms, etc.
        self.root = root
        self.raw = raw
        files = os.listdir(self.root)
        # filter out system files
        self.imgs = [file for file in files if not file.startswith('.') and file.endswith('.jpg')]

    def __len__ (self):
        # Return the total number of images
        return len(self.imgs)

    def __getitem__ (self, index):
        # Read an image at index and perform augmentations
        # Return the tuple : ( augmented tensor , integer label )
        img = Image.open(os.path.join(self.root, self.imgs[index]))
        transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.RandomAffine(
                degrees=20,
                translate=(0, 0.1),
                scale=(0.8, 1.2),
                shear=10
            ),
            tvt.ColorJitter(
                brightness=0.1,
                contrast=0.4,
                saturation=0.3,
                hue=0.2
            ),
            tvt.GaussianBlur(
                kernel_size=5,
                sigma=(0.1,0.2)
            ),
            # tvt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # if needed
        ]) if not self.raw else tvt.ToTensor()

        return (transform(img), random.randint(0, 10))


if __name__ == '__main__':

    # Experimenting with a stop sign
    stop_straight = Image.open("stop_straight.jpg")
    stop_tilt = Image.open("stop_angle_2.jpg")
    # Crop the image:
    target_size = 2048
    stop_straight = stop_straight.crop((590, 502, 590 + 2048,  502 + 2048))
    stop_tilt = stop_tilt.crop((950, 1357, 950 + 2048, 1357 + 2048))
    print("Cropped image shape ", stop_straight.size)
    print("Cropped image shape ", stop_tilt.size)

    # coordinates of reference points on the original image in a circular order:
    #   1____2
    #   /    \
    #  /      \
    #  \      /
    #  4\____/3
    # original (1467, 1806), (1752, 1796), (1739, 1080), (1443, 1087)
    x1, y1 = (855, 590)
    x2, y2 = (1149, 583)
    x3, y3 = (1163, 1301)
    x4, y4 = (879, 1310)
    # tilted   (1828, 2948), (2097, 2712), (2028, 1508), (1717, 1592)
    x1p, y1p = (766, 232)
    x2p, y2p = (1082, 162)
    x3p, y3p = (1156, 1354)
    x4p, y4p = (879, 1609)

    plt.imshow(np.asarray(stop_straight)) # For reference point determination
    # plt.show()
    plt.imshow(np.asarray(stop_tilt)) # For reference point determination
    # plt.show()

    # Solve for h_ij (H):
    R = np.array(
        [[x1, y1, 1, 0, 0, 0, -x1*x1p, -y1*x1p],
        [0, 0, 0, x1, y1, 1, -x1*y1p, -y1*y1p],
        [x2, y2, 1, 0, 0, 0, -x2*x2p, -y2*x2p],
        [0, 0, 0, x2, y2, 1, -x2*y2p, -y2*y2p],
        [x3, y3, 1, 0, 0, 0, -x3*x3p, -y3*x3p],
        [0, 0, 0, x3, y3, 1, -x3*y3p, -y3*y3p],
        [x4, y4, 1, 0, 0, 0, -x4*x4p, -y4*x4p],
        [0, 0, 0, x4, y4, 1, -x4*y4p, -y4*y4p]]
    )

    res = np.append(
        np.linalg.solve(
            R, 
            np.array([x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p])
        ),
        np.array([1])
    )
    H = np.reshape(res, (3, -1))
    # print(H)

    # Convertion derived from direct computation
    startpoints = np.array(
        [[0, 0, 4032, 4032],
        [0, 3024, 3024, 0],
        [1, 1, 1, 1]]
    )
    endpoints = H@startpoints
    endpoints = (endpoints / endpoints[2])[:2]
    # reshape for feeding into tvt.perspective:
    endpoints = endpoints.T
    startpoints = startpoints[:2, :].T
    # print( startpoints.tolist(), endpoints.tolist())
    # print(startpoints.shape, endpoints.shape)

    manual_stop = tvt.functional.perspective(stop_straight, startpoints.tolist(), endpoints.tolist())
    automatic_stop = tvt.functional.perspective(
        stop_straight, 
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        [[x1p, y1p], [x2p, y2p], [x3p, y3p], [x4p, y4p]]
    )

    fig = plt.figure(figsize=(7, 7))
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.title("Original")
    plt.imshow(np.asarray(stop_straight))
    fig.add_subplot(rows, columns, 2)
    plt.title("Target")
    plt.imshow(np.asarray(stop_tilt))
    fig.add_subplot(rows, columns, 3)
    plt.title("Manual")
    plt.imshow(np.asarray(manual_stop))
    fig.add_subplot(rows, columns, 4)
    plt.title("Automatic")
    plt.imshow(np.asarray(automatic_stop))
    # plt.show()

    #################################################################################

    # Play with affine and perspective transforms
    affine = tvt.RandomAffine(
        degrees=20,
        translate=(0, 0.1),
        scale=(0.8, 1.2),
        shear=10
    )
    perspective = tvt.RandomPerspective(
        p=1
    )
    # Play with random affine:
    fig = plt.figure(figsize=(7, 7))
    fig.add_subplot(rows, columns, 1)
    plt.title("Original")
    plt.imshow(np.asarray(stop_straight))
    fig.add_subplot(rows, columns, 2)
    plt.title("First")
    plt.imshow(np.asarray(affine(stop_straight)))
    fig.add_subplot(rows, columns, 3)
    plt.title("Second")
    plt.imshow(np.asarray(affine(stop_straight)))
    fig.add_subplot(rows, columns, 4)
    plt.title("Third")
    plt.imshow(np.asarray(affine(stop_straight)))
    # plt.show()
    
    # Play with random perspective:
    fig = plt.figure(figsize=(7, 7))
    fig.add_subplot(rows, columns, 1)
    plt.title("Original")
    plt.imshow(np.asarray(stop_straight))
    fig.add_subplot(rows, columns, 2)
    plt.title("First")
    plt.imshow(np.asarray(perspective(stop_straight)))
    fig.add_subplot(rows, columns, 3)
    plt.title("Second")
    plt.imshow(np.asarray(perspective(stop_straight)))
    fig.add_subplot(rows, columns, 4)
    plt.title("Third")
    plt.imshow(np.asarray(perspective(stop_straight)))
    # plt.show()

    #################################################################################
    
    # convert for histograms:
    stop_straight = tvt.ToTensor()(stop_straight)
    stop_tilt = tvt.ToTensor()(stop_tilt)
    automatic_stop = tvt.ToTensor()(automatic_stop)

    # add [1:0] so we can ignore 0 that arise from transformations
    hists_stop_straight = [torch.histc(stop_straight[ch],bins=10,min=0,max=1)[1:] for ch in range(3)]
    hists_stop_straight = [hists_stop_straight[ch].div(hists_stop_straight[ch].sum()) for ch in range(3)]

    hists_stop_tilt = [torch.histc(stop_tilt[ch],bins=10,min=0,max=1)[1:] for ch in range(3)]
    hists_stop_tilt = [hists_stop_tilt[ch].div(hists_stop_tilt[ch].sum()) for ch in range(3)]

    hists_automatic_stop = [torch.histc(automatic_stop[ch],bins=10,min=0,max=1)[1:] for ch in range(3)]
    hists_automatic_stop = [hists_automatic_stop[ch].div(hists_automatic_stop[ch].sum()) for ch in range(3)]

    print(hists_automatic_stop)
    
    # Original and target
    print("original and target")
    for ch in range(3):
        dist = wasserstein_distance(
            torch.squeeze(hists_stop_straight[ch]).cpu().numpy(),
            torch.squeeze( hists_stop_tilt[ch] ).cpu().numpy() 
        )
        print("\n Wasserstein distance for channel %d: " % ch, dist)

    # Result and target
    print("result and target")
    for ch in range(3):
        dist = wasserstein_distance(
            torch.squeeze(hists_stop_tilt[ch]).cpu().numpy(),
            torch.squeeze( hists_automatic_stop[ch] ).cpu().numpy() 
        )
        print("\n Wasserstein distance for channel %d: " % ch, dist)

    #################################################################################
    
    PROCESS_IMGS = 1000
    BATCH_SIZE = 16

    my_dataset = MyDataset(ROOT)
    # Helper
    my_dataset_raw = MyDataset(ROOT, raw = True)


    print(len(my_dataset)) # 10
    index = 1
    # print(my_dataset[index][0])
    print (my_dataset[index][0].shape, my_dataset[index][1])
    # torch . Size ([3, 256 , 256 ]) 6
    index = 5
    print (my_dataset[index][0].shape, my_dataset[index][1])
    # torch . Size ([3, 256 , 256 ]) 8

    fig = plt.figure(figsize=(7, 7))
    rows = 3
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(my_dataset_raw[6][0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 2)
    plt.imshow(my_dataset[6][0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 3)
    plt.imshow(my_dataset_raw[8][0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 4)
    plt.imshow(my_dataset[8][0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 5)
    plt.imshow(my_dataset_raw[2][0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 6)
    plt.imshow(my_dataset[2][0].permute(1, 2, 0).numpy())
    # plt.show()


    # We'll also create a sampler, so we can access our images multiple times in a dataloader:
    sampler = torch.utils.data.RandomSampler(my_dataset, replacement=True, num_samples=PROCESS_IMGS)
    my_dataset_gen = torch.utils.data.DataLoader(
        my_dataset, 
        batch_size = 4,
        num_workers = 2,
        # sampler = sampler
    )

    dataloader_out, _ = next(iter(my_dataset_gen))
    print(dataloader_out.shape)

    fig = plt.figure(figsize=(7, 7))
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(dataloader_out[0].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 2)
    plt.imshow(dataloader_out[1].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 3)
    plt.imshow(dataloader_out[2].permute(1, 2, 0).numpy())
    fig.add_subplot(rows, columns, 4)
    plt.imshow(dataloader_out[3].permute(1, 2, 0).numpy())
    # plt.show()

    # Use a smaller test image for this task:
    my_dataset = MyDataset("./imgs_tst")
    my_dataset.imgs =  my_dataset.imgs * 100 # 1000 images total
    
    # Simple dataset
    prev = time.time()
    count = 0
    for i in range(len(my_dataset)):
        _ = my_dataset[i]
        count += 1
    print("Resulting time: ", time.time()-prev, ", total images processed: ", count)

    # Re-initialize dataloader
    my_dataset_gen = torch.utils.data.DataLoader(
        my_dataset, 
        batch_size = 4,
        num_workers = 2,
    )

    # Dataloader
    prev = time.time()
    count = 0
    for test_images, test_labels in my_dataset_gen: 
        _ = test_images
        count += 1
    print("Resulting time: ", time.time()-prev, ", total batches processed: ", count, " with batch size ", BATCH_SIZE)
