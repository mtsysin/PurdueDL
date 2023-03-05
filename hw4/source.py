from ComputationalGraphPrimer import *
import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
import seaborn as sn
import torch
import torch.nn as nn 
import os
import torch.nn.functional as F
from pycocotools.coco import COCO
import shutil
from PIL import Image
import urllib.request
from io import BytesIO
import torchvision.transforms as tvt
import tqdm
from sklearn.metrics import confusion_matrix



ROOT = "."

class COCODataset(torch.utils.data.Dataset):
    def __init__ (self, 
                  root, 
                  categories_list, 
                  num_train = 1500, 
                  num_val = 500, 
                  train = True, 
                  exclusive = True, 
                  clear = False, 
                  transform = None, 
                  augmentation = None,
                  download = False,
                  verify = False, 
                  ):
        super ().__init__()

        # Obtain meta information (e.g. list of file names)
        # Initialize data augmentation transforms, etc.
        self.transform = transform
        self.augmentation = augmentation
        self.root = root
        self.num = num_train if train else num_val
        self.categories_list = categories_list

        if download:
            # Download dataset
 
            # Get COCO objecct
            self.dataType='train2014' if train else 'val2014'
            annFile='{}/annotations/instances_{}.json'.format(self.root,self.dataType)
            if os.path.exists(annFile):
                coco=COCO(annFile)
            else:
                raise ValueError(f"Please download the annotation files into {annFile}")
            self.catIds = coco.getCatIds(catNms=categories_list)

            # If clear, clear the dataset
            if clear:
                if os.path.exists(os.path.join(self.root, "data", "train" if train else "val")):
                    shutil.rmtree(os.path.join(self.root, "data", "train" if train else "val")) 

            # Create necessary folder structure
            for cat in self.categories_list:
                path = os.path.join(self.root, "data", "train" if train else "val", cat)
                if not os.path.exists(path):
                    os.makedirs(path)

            # Download images in the dataset
            for cat in self.categories_list:
                catIds = coco.getCatIds(catNms=cat)
                print("CAT IDs ", catIds)
                imgIds = coco.getImgIds(catIds=catIds)
                print("ids ", len(imgIds))
                random.shuffle(imgIds) # Load in random order
                images = coco.loadImgs(imgIds) 

                count = 0                           # count of downloaded images
                im_iter = iter(images)
                # print(len(images))

                if exclusive:
                    catIds_unacceptable =  [x for x in self.catIds if x != catIds[0]]
                while count < self.num:
                    im = next(im_iter, -1)
                    if im == -1:
                        raise StopIteration("We've ran out of images")
                    # Check for exclusivity:
                    if exclusive:
                        annIds = coco.getAnnIds(imgIds = [im['id']])
                        anns = coco.loadAnns(annIds)
                        not_exclusive = False
                        for ann in anns:
                            if ann["category_id"] in catIds_unacceptable:
                                not_exclusive = True
                                break
                        if not_exclusive: # image has 2 or more categories from the list
                            continue
                    im_pil = self.get_img_from_url(im['coco_url'])
                    save_path = os.path.join(self.root, "data", "train" if train else "val", cat, im['file_name'])
                    im_pil.save(save_path)
                    count += 1
        
        if verify:
            """Verify if the dataset has required number of pictures and that the number is"""
            path = os.path.join(self.root, "data", "train" if train else "val")
            categories_in_folder = [cat for cat in os.listdir(path) if not cat.startswith('.')]
            for cat in tqdm.tqdm(categories_in_folder):
                if cat not in self.categories_list:
                    raise ValueError(f"Unknown category {cat}")
                if len(os.listdir(os.path.join(path, cat))) != self.num:
                    raise ValueError(f"Wrong number of pictures: {len(os.listdir(os.path.join(path, cat)))}")
                if os.path.isfile(os.path.join(path, cat)):
                    raise ValueError("Uncategorized files")
            for cat in tqdm.tqdm(self.categories_list):
                if cat not in os.listdir(path):
                    raise ValueError("Some categories are not downloaded!")
            print("Verification Successful")

        # Now, assuming we have everything downloaded and allocated in folders
        self.path = os.path.join(self.root, "data", "train" if train else "val")
        self.img_dict = {cat: os.listdir(os.path.join(self.path, cat)) for cat in self.categories_list}
        self.cat_idx_encoding = {i: cat for cat, i in zip(self.categories_list, list(range(len(self.categories_list))))}

        # filter out system files
        # self.imgs = [file for file in files if not file.startswith('.') and file.endswith('.jpg')]

    def __len__ (self):
        # Return the total number of images
        return len(self.categories_list) * self.num

    def __getitem__ (self, index):
        # Read an image at index
        # Return the tuple : ( augmented tensor , integer label )
        # Get category:
        cat_idx = index // self.num
        img_index = index % self.num
        category = self.cat_idx_encoding[cat_idx]
        im = Image.open(os.path.join(self.path, category, self.img_dict[category][img_index]))
        if self.transform:
            im = self.transform(im)
            im = im.to(dtype=torch.float64)
        if self.augmentation:
            im = self.augmentation(im)
        return im, cat_idx

    @staticmethod
    def get_img_from_url(url):
        pass
        # Download the image from the URL
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()
        # Resize image:
        im = Image.open(BytesIO(img_data))

        if im.mode != "RGB":
            im = im.convert(mode = "RGB")
        im = im.resize((64,64), Image.BOX)

        return im


class HW4Net(nn.Module):
    """Base neural network form the HW description"""
    def __init__(self, out_classes):
        super(HW4Net, self).__init__() 
        self.conv1 = nn.Conv2d(3, 16, 3)    # Out size 16 * 14 * 14
        self.pool = nn.MaxPool2d(2, 2)      # Out size 16 * 7 * 7
        self.conv2 = nn.Conv2d(16, 32, 3)   # Out size 32 * 5 * 5
        self.fc1 = nn.Linear(32 * 14 * 14, 64) # Add actual value
        self.fc2 = nn.Linear(64, out_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Out size 16 * 62 * 62 -> # Out size 16 * 31 * 31
        x = self.pool(F.relu(self.conv2(x)))  # Out size 32 * 29 * 29 -> # Out size 32 * 14 * 14
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class HW4Net_v2(nn.Module): 
    """Modification: added padding to the convolutional layers"""
    def __init__(self, out_classes):
        super(HW4Net_v2, self).__init__() 
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64) # Add actual value
        self.fc2 = nn.Linear(64, out_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Out size 16 * 62 * 62 -> # Out size 16 * 31 * 31
        x = self.pool(F.relu(self.conv2(x)))  # Out size 32 * 29 * 29 -> # Out size 32 * 14 * 14
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        

class HW4Net_v3(nn.Module): 
    """Adding more convo layers + max pool"""
    def __init__(self, out_classes, num_extra_layers):
        self.num_extra_layers = num_extra_layers
        super(HW4Net_v3, self).__init__() 
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding = 1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64) # Add actual value
        self.fc2 = nn.Linear(64, out_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Out size 16 * 64 * 64 -> # Out size 16 * 32 * 32
        x = self.pool(F.relu(self.conv2(x)))  # Out size 32 * 32 * 32 -> # Out size 32 * 16 * 16
        for _ in range(self.num_extra_layers):
            x = F.relu(self.conv3(x))  # Out size 32 * 16 * 16
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(net, save = False):
    # Choose device
    if torch.cuda.is_available()== True: 
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")

    net.train()

    # Create transform
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch = 10

    train_dataset = COCODataset(
        root=ROOT,
        categories_list=['airplane', 'bus', 'cat', 'dog', 'pizza'],
        num_train=1500,
        num_val=500,
        download = False, 
        verify = True, 
        train = True, 
        transform=transform
    )

    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                                  batch_size = batch, 
                                                  shuffle = True, 
                                                  num_workers = 0)

    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=1e-3, 
        betas=(0.9, 0.99)
    )
    
    losses = []

    epochs = 10

    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
    outer = tqdm.tqdm(total=epochs, desc='Epochs', position=0)
    for epoch in range(epochs):
        running_loss = 0.0
        inner = tqdm.tqdm(total=len(train_data_loader), desc='Batches', position=0)
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100)
                )
                losses.append(running_loss / 100)
                running_loss = 0.0

            inner.update(1)
        outer.update(1)

    if save:
        torch.save(net.state_dict(), ROOT+'/model')

    return losses

def val(net):
    # Choose device
    if torch.cuda.is_available()== True: 
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")

    net.eval()

    # Create transform
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch = 10

    val_dataset = COCODataset(
        root=ROOT,
        categories_list=['airplane', 'bus', 'cat', 'dog', 'pizza'],
        num_train=1500,
        num_val=500,
        download = False, 
        verify = True, 
        train = False, 
        transform=transform
    )

    val_data_loader = torch.utils.data.DataLoader(dataset = val_dataset, 
                                                batch_size = batch, 
                                                shuffle = True, 
                                                num_workers = 0)

    test_loss, correct = 0, 0
    criterion = torch.nn.CrossEntropyLoss() 
    size = len(val_data_loader.dataset)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(val_data_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device) 
            outputs = net(inputs)
            test_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            pred_labels.extend(outputs.argmax(1).view(-1).numpy())
            true_labels.extend(labels.view(-1).numpy())
    test_loss /= size #batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    labels = ['airplane', 'bus', 'cat', 'dog', 'pizza']

    return confusion_matrix(true_labels, pred_labels)
    
    
    
if __name__=="__main__":
    # Initialization
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] =  str(seed)

    # dataset = COCODataset(
    #     root=ROOT,
    #     categories_list=['airplane', 'bus', 'cat', 'dog', 'pizza'],
    #     num_train=1500,
    #     num_val=500,
    #     download = False, 
    #     verify = True, 
    #     train = True, 
    # )

    dataset = COCODataset(
        root=ROOT,
        categories_list=['airplane', 'bus', 'cat', 'dog', 'pizza'],
        num_train=1500,
        num_val=500,
        download = False, 
        verify = True, 
        train = False, 
    )

    # Test dataset output, note that no transform was initialized for the dataset output
    classes = 5
    img_per_calss = 5
    f, axarr = plt.subplots(classes,img_per_calss)
    for cat in range(classes):
        for i in range(img_per_calss):
            img_index = random.randint(0, dataset.num) + dataset.num * cat
            axarr[cat,i].imshow(dataset[img_index][0])
            axarr[cat,i].axis('off')
    plt.savefig("./out/data_val.png")


    net1 = HW4Net(5)
    net2 = HW4Net_v2(5)
    net3 = HW4Net_v3(5, 10)

    # summary(net1, input_size=(10, 3, 64, 64))
    # summary(net2, input_size=(10, 3, 64, 64))
    # summary(net3, input_size=(10, 3, 64, 64))

    # net1 = net1.to(torch.float64)
    # net2 = net2.to(torch.float64)
    # net3 = net3.to(torch.float64)

    # loss_trace1 = train(net1)
    # plt.plot(loss_trace1)
    # plt.ylabel('Loss')
    # plt.xlabel('Processed batches * 100')
    # plt.savefig("./out/loss_trace1.png")

    # loss_trace2 = train(net2)
    # plt.plot(loss_trace2)
    # plt.ylabel('Loss')
    # plt.xlabel('Processed batches * 100')
    # plt.savefig("./out/loss_trace2.png")

    # loss_trace3 = train(net3)
    # plt.plot(loss_trace3)
    # plt.ylabel('Loss')
    # plt.xlabel('Processed batches * 100')
    # plt.savefig("./out/loss_trace3.png")

    # labels = ['airplane', 'bus', 'cat', 'dog', 'pizza']

    # cm1 = val(net1)
    # plt.figure(figsize = (12,7))
    # hm = sn.heatmap(data=cm1,
    #             annot=True,
    #             xticklabels=labels, 
    #             yticklabels=labels,
    #             square=1, 
    #             linewidth=1.,
    #             fmt = '.0f'
    #         )
    # plt.savefig("./out/hm1.png")

    # cm2 = val(net2)
    # plt.figure(figsize = (12,7))
    # hm = sn.heatmap(data=cm2,
    #             annot=True,
    #             xticklabels=labels, 
    #             yticklabels=labels,
    #             square=1, 
    #             linewidth=1.,
    #             fmt = '.0f'
    #         )
    # plt.savefig("./out/hm2.png")

    # cm3 = val(net3)
    # plt.figure(figsize = (12,7))
    # hm = sn.heatmap(data=cm3,
    #             annot=True,
    #             xticklabels=labels, 
    #             yticklabels=labels,
    #             square=1, 
    #             linewidth=1.,
    #             fmt = '.0f'
    #         )
    # plt.savefig("./out/hm3.png")


