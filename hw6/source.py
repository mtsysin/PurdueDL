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
from dataset import COCODataset
from loss import YOLOLoss
from typing import Any, Callable, List, Optional, Type, Union
from model import ResnetBlock, HW5Net
from operator import add

MIN_W = 200
MIN_H = 200
ROOT = "."
LOSS_COUNT = 50

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
        download = False,
        verify = True,
        train = True,
        transform=transform,
    )

    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                                  batch_size = batch, 
                                                  shuffle = True, 
                                                  num_workers = 0)

    net = net.to(device)

    criterion = YOLOLoss()
    # criterion_localization = torch.nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=1e-3, 
        betas=(0.9, 0.99)
    )
    
    losses = []
    losses_separate = []

    epochs = 5

    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
    outer = tqdm.tqdm(total=epochs, desc='Epochs', position=0)
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_separate = [0.0] * 3
        inner = tqdm.tqdm(total=len(train_data_loader), desc='Batches', position=0)
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)

            # print(inputs[0, 0, 0,...])

            labels = labels.to(device) 

            optimizer.zero_grad()

            outputs = net(inputs)

            # print(outputs[0, 0, 0,...])

            batch_losses = criterion(outputs, labels)
            # print(batch_losses)

            loss = sum(batch_losses)
            # print(loss)

            loss.backward()
            # for param in net.parameters():
                # print(param.grad[0,...], param.size())
            #     break
            optimizer.step()

            running_loss += loss.item()
            running_loss_separate = [curr_loss + new_loss.item() for curr_loss, new_loss in zip(running_loss_separate, batch_losses)]


            if (i+1) % LOSS_COUNT == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / LOSS_COUNT)
                )
                losses.append(running_loss / LOSS_COUNT)
                losses_separate.append([el/LOSS_COUNT for el in running_loss_separate])
                running_loss = 0.0
                running_loss_separate = [0.0] * 3

                # print("Labels bboxes", labels_bboxes)
                # print("Labels classes", labels_classes)
                # print("OUT bboxes", output_bboxes)
                # print("OUT classes", output_classes)

            inner.update(1)
        outer.update(1)

    if save:
        torch.save(net.state_dict(), ROOT+'/model')

    return losses, losses_separate

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

    class_list = ['bus', 'cat', 'pizza']

    net = HW5Net(3)
    net = net.to(torch.float32)
    # net.load_state_dict(torch.load(ROOT+'/model', map_location=torch.device('cpu')))

    loss_trace, loss_traces = train(net, save=True)
    plt.plot(loss_trace)
    for trace in zip(*loss_traces):
        plt.plot(trace)

    plt.ylabel('Loss')
    plt.xlabel('Processed batches * 100')
    plt.savefig("./out/loss_trace1.png")


    # cm1 = val(net, load_path=ROOT+'/model')
    # plt.figure(figsize = (12,7))
    # hm = sn.heatmap(data=cm1,
    #     annot=True,
    #     xticklabels=class_list, 
    #     yticklabels=class_list,
    #     square=1, 
    #     linewidth=1.,
    #     fmt = '.0f'
    # )
    # plt.savefig("./out/hm.png")



    # dataset = COCODataset(
    #     root=ROOT,
    #     categories_list=class_list,
    #     num_train_min=100,
    #     num_train_max=4200,
    #     num_val_min=1000,
    #     num_val_max=3000,
    #     download = False,
    #     verify = True,
    #     train = False,
    #     # clear=True
    # )

    # transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # val_dataset = COCODataset(
    #     root=ROOT,
    #     categories_list=class_list,
    #     num_train_min=2000,
    #     num_train_max=4000,
    #     num_val_min=1000,
    #     num_val_max=3000,
    #     download = False,
    #     verify = True,
    #     train = False,
    #     transform=transform
    # )

    for _ in range(0):

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
