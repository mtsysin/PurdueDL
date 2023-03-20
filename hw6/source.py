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
    # seed = 0
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] =  str(seed)

    class_list = ['bus', 'cat', 'pizza']

    net = HW5Net(3)
    net = net.to(torch.float32)
    net.load_state_dict(torch.load(ROOT+'/models/model', map_location=torch.device('cpu')))

    # loss_trace, loss_traces = train(net, save=True)
    # labels_names = ["BCE", "MSE", "Cross entropy"]
    # plt.plot(loss_trace, label = "Combined")
    # for i, trace in enumerate(zip(*loss_traces)):
    #     plt.plot(trace, label = labels_names[i])

    # plt.ylabel('Loss')
    # plt.xlabel('Processed batches * 100')
    # plt.legend()
    # plt.savefig("./out/loss_trace1.png")

    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        train = False,
        clear = False,
        download = False,
        verify = True,
        grid_size = 32,
        return_raw = False,
        anchor_boxes = 5,
        transform=transform
    )
    raw_dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        train = False,
        clear = False,
        download = False,
        verify = True,
        grid_size = 32,
        return_raw = False,
        anchor_boxes = 5,
    )

    MAX_SHOW = 3

    for _ in range(1):
        # Get index from the dataset
        idx =  np.random.randint(0, len(dataset)) # 452, 2343
        print(idx)
        image, _ = raw_dataset[idx]
        image_input , label = dataset[idx]

        label = net(image_input.unsqueeze(0)).squeeze(0)

        print("Image size: ", image.size)
        print("Label size: ", label.size())
        # print(label[..., 0])

        # Find indices and bboxes where there is an image:
        pred_bce = nn.Sigmoid()(label[..., 0])
        # top = torch.topk(pred_bce, 3, dim=-1)
        # print("TOP: ", top)
        Iobj_i = (nn.Sigmoid()(label[..., 0])>0.01).bool()

        selected_igms = label[Iobj_i]
        _, select_ind = torch.topk(selected_igms[..., 0], 2)
        print(select_ind)
        selected_igms = selected_igms[select_ind]
        selected_igms_positions = Iobj_i.nonzero(as_tuple=False)
        selected_igms_positions =selected_igms_positions[select_ind]
        print("Selected yolo vectors: ", selected_igms)
        print("Selected yolo vectors positions: ", selected_igms_positions)

        # Show image and corresponding bounding boxes:
        image = np.uint8(image)
        fig, ax = plt.subplots(1, 1)
        for yolo_vector, position in zip(selected_igms, selected_igms_positions):
            # get bbox values and convert them to scalars
            x, y, w, h = yolo_vector[1:5].tolist()
            class_vector = yolo_vector[5:].tolist()
            class_index = class_vector.index(max(class_vector))
            anchor_idx, x_idx, y_idx = position.tolist()

            if anchor_idx == 0: 
                w_scale, h_scale = 3, 1
            if anchor_idx == 1: 
                w_scale, h_scale = 2, 1
            if anchor_idx == 2: 
                w_scale, h_scale = 1, 1
            if anchor_idx == 3: 
                w_scale, h_scale = 1, 2
            if anchor_idx == 4: 
                w_scale, h_scale = 1, 3

            # Select correct dimenstions
            x = int((x + (x_idx + 0.5)) * dataset.grid_size)
            y = int((y + (y_idx + 0.5)) * dataset.grid_size)
            w = int(math.exp(w) * dataset.grid_size * w_scale)
            h = int(math.exp(h) * dataset.grid_size * h_scale)

            print(x, y, w, h)

            image = cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (36, 255, 12), 2) 
            image = cv2.putText(image, class_list[class_index], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (36, 255, 12), 2)


        ax.imshow(image) 
        ax.set_axis_off() 
        plt.axis('tight') 
        plt.show()

