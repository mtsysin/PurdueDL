# from ComputationalGraphPrimer import *
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
import torchvision
import tqdm
from sklearn.metrics import confusion_matrix
import json 
import cv2

MIN_W = 200
MIN_H = 200
ROOT = "."

class COCODataset(torch.utils.data.Dataset):
    """Iteration of dataset for HW4"""
    def __init__ (self, 
                  root, 
                  categories_list, 
                  num_train_min = 1500, 
                  num_train_max = 2000,
                  num_val_min = 500, 
                  num_val_max = 500, 
                  train = True, 
                  exclusive = True, 
                  clear = False, 
                  transform = None, 
                  scale_bbox = False,
                  augmentation = None,
                  download = False,
                  verify = False,
                  ):
        super ().__init__()

        # Obtain meta information (e.g. list of file names)
        # Initialize data augmentation transforms, etc.
        self.transform = transform
        self.scale_bbox = scale_bbox
        self.augmentation = augmentation
        self.root = root
        self.num_min = num_train_min if train else num_val_min
        self.num_max = num_train_max if train else num_val_max
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

            self.catIds_to_category = {cat_id: i for i, cat_id in enumerate(self.catIds)}

            # If clear, clear the dataset
            if clear:
                path = os.path.join(self.root, "data", "train" if train else "val")
                if os.path.exists(path):
                    shutil.rmtree(path) 
                if not os.path.exists(path):
                    os.makedirs(path)

            # Download images in the dataset
            print("CAT IDs ", self.catIds)
            imgIds = list(set(sum([coco.getImgIds(catIds=cat_id) for cat_id in self.catIds], [])))
            # imgIds=coco.getImgIds(catIds=self.catIds)
            print("img_ids ", len(imgIds))
            random.shuffle(imgIds) # Load in random order
            images = coco.loadImgs(imgIds) 

            count = 0                           # count of downloaded images
            im_iter = iter(images)
            print(len(images))

            # Create custom lightweight annotation file.
            self.annotation = {}
            # Download status bar
            status_bar = tqdm.tqdm(total=self.num_max, desc='LOADED IMAGES', position=0)
            status_bar_total_img = tqdm.tqdm(total=len(images), desc='PROCESSED IMAGES', position=1)

            while count < self.num_max:
                im = next(im_iter, -1)
                # Check if there is something in the iterator
                if im == -1 and count < self.num_min:
                    raise StopIteration("We've ran out of images, target not reached")
                elif im == -1:
                    break
                # Get annotations for objects
                annIds = coco.getAnnIds(imgIds = [im['id']])
                anns = coco.loadAnns(annIds)

                max_bbox_area = 0
                max_box_area_cat_id = None
                max_bbox = None
                # Check if there is a dominant object:
                for ann in anns:
                    # bbox_area = ann["bbox"][2] * ann["bbox"][3] # W * H
                    bbox_area = ann["area"] # W * H
                    # print(ann["bbox"], ann["area"])
                    # assert ann["bbox"][2] * ann["bbox"][3] >= ann["area"]
                    if bbox_area > max_bbox_area:
                        max_bbox_area = bbox_area
                        max_box_area_cat_id = ann["category_id"]
                        max_bbox = ann["bbox"]
                    # if ann["category_id"] in self.catIds:
                    #     img_annotation[ann["category_id"]] = ann["bbox"]
                img_annotation = [max_box_area_cat_id, max_bbox]
                        
                # Accept box if the bounding box is maximal, big enough, and comes from correct category
                if max_bbox[2] * max_bbox[3] > MIN_W * MIN_H and max_box_area_cat_id in self.catIds: ############# CHANGES

                    im_pil, orig_shape = self.get_img_from_url(im['coco_url'])
                    save_path = os.path.join(self.root, "data", "train" if train else "val", str(im['id'])+".jpg")
                    im_pil.save(save_path)

                    w, h = orig_shape
                    img_annotation[1][0] = img_annotation[1][0] * 256 // w
                    img_annotation[1][1] = img_annotation[1][1] * 256 // h
                    img_annotation[1][2] = img_annotation[1][2] * 256 // w
                    img_annotation[1][3] = img_annotation[1][3] * 256 // h

                    img_annotation[0] = self.catIds_to_category[img_annotation[0]]

                    self.annotation[im['id']] = img_annotation
                    count += 1
                    status_bar.update(1)
                status_bar_total_img.update(1)
            
            print(f"Loaded {count} images")
            with open(os.path.join(self.root, "data", "data_" + ("train" if train else "val") + ".json"), "w") as outfile:
                json.dump(self.annotation, outfile)
        
        if verify:
            """Verify if the dataset has required number of pictures and that the number is"""
            path = os.path.join(self.root, "data", "train" if train else "val")
            images_in_folder = [cat for cat in os.listdir(path) if not cat.startswith('.')]
            if len(images_in_folder) < self.num_min or len(images_in_folder) > self.num_max:
                    raise ValueError(f"Wrong number of pictures: {len(images_in_folder)}")
            for img in images_in_folder:
                if not os.path.isfile(os.path.join(path, img)):
                    raise ValueError("Sub-folders present")
            print("Verification Successful")

        # Now, assuming we have everything downloaded and allocated in folders
        self.path = os.path.join(self.root, "data", "train" if train else "val")
        annot_file_name = "data_" + ("train" if train else "val") + ".json"
        with open(os.path.join(self.root, "data", annot_file_name), "r") as annot_file:
            self.annotation =  json.load(annot_file)
        self.img_list = os.listdir(os.path.join(self.path))
        self.img_list = [file.split('.')[0] for file in self.img_list if not file.startswith('.') and file.endswith('.jpg')]

    def __len__ (self):
        # Return the total number of images
        return len(self.img_list)

    def __getitem__ (self, index):
        # Read an image at index
        # Return the tuple : ( augmented tensor , integer label )
        # Get category:
        img_index = self.img_list[index]
        category, bbox = self.annotation[img_index]
        bbox = torch.tensor(bbox).double()
        im = Image.open(os.path.join(self.path, img_index + '.jpg'))
        if self.transform:
            im = self.transform(im)
            im = im.to(dtype=torch.float64)
        if self.augmentation:
            im = self.augmentation(im)
        if self.scale_bbox:
            bbox /= 256.0
        return im, category, bbox
    
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
        size = im.size
        im = im.resize((256,256), Image.BOX)

        return im, size
        

class ResnetBlock(nn.Module):
    """
    Inspired by the original implementation in pytorch github
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
    ) -> None:
        super().__init__()
        
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = norm_layer(outplanes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

IM_SIZE = 256

class HW5Net(nn.Module):
    """
    Resnet-based encoder that consists of a few downsampling + several Resnet blocks as the backbone and two prediction heads.
    """
    def __init__(self, input_nc, output_nc, ngf=8, n_blocks=4, classes=3):
        """ Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images)
        ngf (int) -- the number of filters first conv layer
        n_blocks (int) -- teh number of ResNet blocks
        """
        assert (n_blocks >= 0) 
        super(HW5Net, self).__init__() 
        # The first conv layer
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]
        # Add downsampling layers
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ] 
        # Add your own ResNet blocks
        mult = 2 ** n_downsampling 
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, ngf * mult)] 

        # Generate final model
        self.model = nn.Sequential(*model) 
        # The classification head 
        class_head = [
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(ngf * IM_SIZE * IM_SIZE // mult // 4, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(64, classes)
        ]
        self.class_head = nn.Sequential(*class_head) # The bounding box regression head
        # Localization head
        bbox_head = [
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(ngf * IM_SIZE * IM_SIZE // mult // 4, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 4)
        ]
        self.bbox_head = nn.Sequential(*bbox_head)

    def forward(self, input):
        ft = self.model(input) 
        cls = self.class_head(ft) 
        bbox = self.bbox_head(ft) 
        return cls, bbox
    

class LocLoss(nn.Module):
    def __init__(self, reduction = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, true_box, pred_box):
        true = torch.cat((true_box[..., :2], true_box[..., :2] + true_box[..., 2:4]), -1)
        pred = torch.cat((pred_box[..., :2], pred_box[..., :2] + pred_box[..., 2:4]), -1)
        loss = torchvision.ops.complete_box_iou_loss(true, pred, self.reduction)

        # print(loss)
        return loss
    

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
        categories_list=['bus', 'cat', 'pizza'],
        num_train_min=2000,
        num_train_max=4000,
        num_val_min=1000,
        num_val_max=3000,
        download = False,
        verify = True,
        train = True,
        transform=transform,
        scale_bbox = True,
    )

    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                                  batch_size = batch, 
                                                  shuffle = True, 
                                                  num_workers = 0)

    net = net.to(device)

    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_localization = LocLoss()
    # criterion_localization = torch.nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=1e-3, 
        betas=(0.9, 0.99)
    )
    
    losses = []
    losses_class = []
    losses_loc = []

    epochs = 5

    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
    outer = tqdm.tqdm(total=epochs, desc='Epochs', position=0)
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_class = 0.0
        running_loss_loc = 0.0
        inner = tqdm.tqdm(total=len(train_data_loader), desc='Batches', position=0)
        for i, data in enumerate(train_data_loader):
            inputs, labels_classes, labels_bboxes = data
            inputs = inputs.to(device)
            labels_classes = labels_classes.to(device) 
            labels_bboxes = labels_bboxes.to(device) 

            optimizer.zero_grad()

            output_classes, output_bboxes = net(inputs)

            loss_class = criterion_class(output_classes, labels_classes) 
            loss_loc = criterion_localization(output_bboxes, labels_bboxes)
            loss = loss_class + loss_loc
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_class += loss_class.item()
            running_loss_loc += loss_loc.item()

            if (i+1) % 100 == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100)
                )
                losses.append(running_loss / 100)
                losses_class.append(running_loss_class/100)
                losses_loc.append(running_loss_loc/100)
                running_loss = 0.0
                running_loss_class = 0.0
                running_loss_loc = 0.0


                # print("Labels bboxes", labels_bboxes)
                # print("Labels classes", labels_classes)
                # print("OUT bboxes", output_bboxes)
                # print("OUT classes", output_classes)

            inner.update(1)
        outer.update(1)

    if save:
        torch.save(net.state_dict(), ROOT+'/model')

    return losses, losses_class, losses_loc

def iou(true_box, pred_box):
    true = torch.cat((true_box[..., :2], true_box[..., :2] + true_box[..., 2:4]), -1).unsqueeze(0)
    pred = torch.cat((pred_box[..., :2], pred_box[..., :2] + pred_box[..., 2:4]), -1).unsqueeze(0)
    return torchvision.ops.box_iou(true, pred)[0, 0]

def val(net, load_path=None):
    # Choose device
    if torch.cuda.is_available() == True: 
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")

    if load_path:
        net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    net.eval()

    # Create transform
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch = 10

    val_dataset = COCODataset(
        root=ROOT,
        categories_list=['bus', 'cat', 'pizza'],
        num_train_min=2000,
        num_train_max=4000,
        num_val_min=1000,
        num_val_max=3000,
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
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_localization = LocLoss()
    size = len(val_data_loader.dataset)

    true_labels = []
    pred_labels = []

    test_iou = 0

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(val_data_loader)):

            inputs, labels_classes, labels_bboxes = data
            inputs = inputs.to(device)
            labels_classes = labels_classes.to(device) 
            labels_bboxes = labels_bboxes.to(device) 

            # inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device) 

            # outputs = net(inputs)

            output_classes, output_bboxes = net(inputs)

            for i in range(output_bboxes.size()[0]):
                test_iou += iou(output_bboxes[i, ...], labels_bboxes[i, ...])
                print(iou(output_bboxes[i, ...], labels_bboxes[i, ...]))

            test_loss += criterion_class(output_classes, labels_classes).item() + \
                criterion_localization(output_bboxes, labels_bboxes).item()

            correct += (output_classes.argmax(1) == labels_classes).type(torch.float).sum().item()
            pred_labels.extend(output_classes.argmax(1).view(-1).numpy())
            true_labels.extend(labels_classes.view(-1).numpy())

    test_iou /= size
    test_loss /= size #batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test IOU Error: {test_iou}\n")


    return confusion_matrix(true_labels, pred_labels)
    
    
    
if __name__=="__main__":
    # Initialization
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] =  str(seed)

    # test IOU:
    # print(iou(
    #     torch.tensor([2.5, 2.5, 1, 1]),
    #     torch.tensor([2, 2, 3, 3]),
    # ))
    # raise ValueError()

    # Verify NN structure 
    # model = HW5Net(3, 1)
    # test = torch.rand(16, 3, 256, 256)
    # summary(model, input_size=(16, 3, 256, 256))
    # print(len(list(model.parameters())))
    # print(model(test)[0].size(), model(test)[1].size())

    class_list = ['bus', 'cat', 'pizza']

    dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        num_train_min=100,
        num_train_max=4200,
        num_val_min=1000,
        num_val_max=3000,

        download = False,
        verify = True,
        train = True,
        # clear=True
    )

    for _ in range(0):
        idx = np.random.randint(0, len(dataset))

        image, c, bbox = dataset[idx]
        print(c, bbox)
        print(image.size)
        [x, y, w, h] = bbox
        label = c

        image = np.uint8(image)
        fig, ax = plt.subplots(1, 1)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (36, 255, 12), 2) 
        image = cv2.putText(image, class_list[label], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (36, 255, 12), 2)

        ax.imshow(image) 
        ax.set_axis_off() 
        plt.axis('tight') 
        plt.show()

    net = HW5Net(3, 1)
    # summary(net, input_size=(16, 3, 256, 256))
    net = net.to(torch.float64)
    net.load_state_dict(torch.load(ROOT+'/model', map_location=torch.device('cpu')))

    # loss_trace, loss_trace_class, loss_trace_loc = train(net, save=True)
    # plt.plot(loss_trace)
    # plt.plot(loss_trace_class)
    # plt.plot(loss_trace_loc)

    # plt.ylabel('Loss')
    # plt.xlabel('Processed batches * 100')
    # plt.savefig("./out/loss_trace1.png")


    cm1 = val(net, load_path=ROOT+'/model')
    plt.figure(figsize = (12,7))
    hm = sn.heatmap(data=cm1,
        annot=True,
        xticklabels=class_list, 
        yticklabels=class_list,
        square=1, 
        linewidth=1.,
        fmt = '.0f'
    )
    plt.savefig("./out/hm.png")



    dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        num_train_min=100,
        num_train_max=4200,
        num_val_min=1000,
        num_val_max=3000,
        download = False,
        verify = True,
        train = False,
        # clear=True
    )

    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        num_train_min=2000,
        num_train_max=4000,
        num_val_min=1000,
        num_val_max=3000,
        download = False,
        verify = True,
        train = False,
        transform=transform
    )

    for _ in range(1):

        idx = np.random.randint(0, len(dataset))
        image, c, bbox = dataset[idx-103]
        val_image, _, _ = val_dataset[idx-103]

        print(c, bbox)
        print(image.size)

        [x, y, w, h] = bbox
        label = c

        print(val_image.size())
        val_image = val_image.unsqueeze(0)
        print(val_image.size())
        # print(val_image)

 
        output_classes, output_bboxes = net(val_image)
        print(output_bboxes)

        print(output_classes.size())
        print(output_bboxes.size())

        print(output_classes[0, ...].size())
        print(output_bboxes[0, ...].size())


        [x_p, y_p, w_p, h_p] = (output_bboxes[0, ...]*256).to(dtype=torch.int).tolist()
        print(x_p, y_p, w_p, h_p)
        print(x, y, w, h)

        out_class = output_classes[0].argmax(0)
        print("out", out_class)

        image = np.uint8(image)
        fig, ax = plt.subplots(1, 1)
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (36, 255, 12), 2) 
        image = cv2.rectangle(image, (int(x_p), int(y_p)), (int(x_p + w_p), int(y_p + h_p)), (233, 9, 12), 2) 

        image = cv2.putText(image, class_list[label] + " -> " + class_list[out_class], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (36, 255, 12), 2)

        ax.imshow(image) 
        ax.set_axis_off() 
        plt.axis('tight') 
        plt.show()
