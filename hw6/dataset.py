# from ComputationalGraphPrimer import *
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
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
import json 
import cv2
import math

MIN_W = 64
MIN_H = 64
TARGET_SIZE = 265
ROOT = "."


class COCODataset(torch.utils.data.Dataset):
    """Iteration of dataset for HW4"""
    def __init__ (self, 
                root, 
                categories_list, 
                num_train_min = 4500, 
                num_train_max = 7000,
                num_val_min = 2500, 
                num_val_max = 4000, 
                train = True, 
                clear = False, 
                transform = None, 
                augmentation = None,
                download = False,
                verify = False,
                grid_size = 32,
                return_raw = False,
                anchor_boxes = 5
        ):
        super ().__init__()

        # Obtain meta information (e.g. list of file names)
        # Initialize data augmentation transforms, etc.
        self.transform = transform
        self.augmentation = augmentation
        self.root = root
        self.num_min = num_train_min if train else num_val_min
        self.num_max = num_train_max if train else num_val_max
        self.categories_list = categories_list
        self.grid_size = grid_size
        self.return_raw = return_raw
        self.anchor_boxes = anchor_boxes

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
            print("Number of unfiltered images: ", len(images))

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
                anns = coco.loadAnns(annIds) # annotation for particular image

                max_bbox_area = 0
                max_box_area_cat_id = None
                max_bbox = None
                img_annotation = []
                # Check if there is a dominant object:
                for ann in anns:
                    # bbox_area = ann["bbox"][2] * ann["bbox"][3] # W * H
                    # print(ann["bbox"], ann["area"])
                    obj_area = ann["area"] # W * H
                    obj_cat_id = ann["category_id"]
                    if obj_area > MIN_W * MIN_H and obj_cat_id in self.catIds:
                        img_annotation.append([obj_cat_id, ann["bbox"]])
                        
                # Accept image if there is something in annotations:
                if img_annotation: ############# CHANGES

                    im_pil, orig_shape = self.get_img_from_url(im['coco_url'])
                    save_path = os.path.join(self.root, "data", "train" if train else "val", str(im['id'])+".jpg")
                    im_pil.save(save_path)

                    w, h = orig_shape
                    # Scale bounding boxes accordingly:
                    img_annotation = [
                        [
                            self.catIds_to_category[ann[0]], 
                            [
                                ann[1][0] * TARGET_SIZE // w,
                                ann[1][1] * TARGET_SIZE // h,
                                ann[1][2] * TARGET_SIZE // w,
                                ann[1][3] * TARGET_SIZE // h,
                            ]
                        ] 
                        for ann in img_annotation
                    ]

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
        # Remove clutter
        self.img_list = [file.split('.')[0] for file in self.img_list if not file.startswith('.') and file.endswith('.jpg')]

    def __len__ (self):
        # Return the total number of images
        return len(self.img_list)

    def __getitem__ (self, index):
        # Read an image at index
        # Return the tuple : ( augmented tensor , integer label )
        # Get category:
        img_index = self.img_list[index]
        im = Image.open(os.path.join(self.path, img_index + '.jpg'))
        if self.transform:
            im = self.transform(im)
            im = im.to(dtype=torch.float32)
        if self.augmentation:
            im = self.augmentation(im)
        annotations = self.annotation[img_index]
        
        # If return raw, return it
        if self.return_raw:
            return im, annotations
        
        # bounding box [top left x position , top left y position , width, height]
        #Iterate through annotations to generate a yolo tensor:
        S = TARGET_SIZE // self.grid_size
        A = self.anchor_boxes
        C = len(self.categories_list)
        label = torch.zeros((A, S, S, 5 + C))
        for ann in annotations:
            # Determine the correct box index
            # Get coordiantes:
            x_tl, y_tl, w, h = ann[1]
            x_center, y_center = x_tl + w / 2.0, y_tl + h / 2.0
            cell_x_idx = int(min(S-1, x_center // self.grid_size))
            cell_y_idx = int(min(S-1, y_center // self.grid_size))

            # select anchor box
            w_scale, h_scale = w / self.grid_size, h / self.grid_size
            AR = h / w
            if AR <= 0.2: 
                abox_idx = 0
                w_scale, h_scale = w / 3, h
            if 0.2 < AR <= 0.5: 
                abox_idx = 1
                w_scale, h_scale = w / 2, h
            if 0.5 < AR <= 1.5: 
                abox_idx = 2
                w_scale, h_scale = w, h
            if 1.5 < AR <= 4.0: 
                abox_idx = 3
                w_scale, h_scale = w, h / 2
            if 4.0 < AR: 
                abox_idx = 4
                w_scale, h_scale = w, h / 3

            bbox_scaled = [
                x_center/self.grid_size - (cell_x_idx + 0.5),
                y_center/self.grid_size - (cell_y_idx + 0.5),
                math.log(w_scale/self.grid_size),
                math.log(h_scale/self.grid_size)
            ]

            yolo_vector = torch.FloatTensor([1] + bbox_scaled + [0] * C)

            if label[abox_idx, cell_x_idx, cell_y_idx, 0] == 0:
                label[abox_idx, cell_x_idx, cell_y_idx, ...] = yolo_vector
                # Set objectness
                label[abox_idx, cell_x_idx, cell_y_idx, 0] = 1
                # Set class
                label[abox_idx, cell_x_idx, cell_y_idx, 5 + ann[0]] = 1

        return im, label.float()
    
    
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
    
if __name__ == "__main__":
    """
    Test dataset functionality
    """

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] =  str(seed)

    class_list = ['bus', 'cat', 'pizza']

    # Dataset downloader

    # dataset = COCODataset(
    #     root=ROOT,
    #     categories_list=class_list,
    #     num_train_min = 4500, 
    #     num_train_max = 7000,
    #     num_val_min = 2500, 
    #     num_val_max = 4000,
    #     train = False,
    #     clear = True,
    #     download = True,
    #     verify = True
    # )

    # Verify that dataset works correctly:

    dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        train = True,
        clear = False,
        download = False,
        verify = True,
        grid_size = 32,
        return_raw = False,
        anchor_boxes = 5
    )

    for _ in range(1):
        # Get index from the dataset
        idx = 25 #np.random.randint(0, len(dataset))
        image, label = dataset[idx]

        print("Image size: ", image.size)
        print("Label size: ", label.size())

        # Find indices and bboxes where there is an image:
        Iobj_i = label[..., 0].bool()

        selected_igms = label[Iobj_i]
        selected_igms_positions = Iobj_i.nonzero(as_tuple=False)
        print("Selected yolo vectors: ", selected_igms)
        print("Selected yolo vectors positions: ", selected_igms_positions)

        # Show image and corresponding bounding boxes:
        image = np.uint8(image)
        fig, ax = plt.subplots(1, 1)
        for yolo_vector, position in zip(selected_igms, selected_igms_positions):
            # get bbox values and convert them to scalars
            x, y, w, h = yolo_vector[1:5].tolist()
            class_vector = yolo_vector[5:].tolist()
            class_index = class_vector.index(1.0)
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

    
    # Raw
    dataset = COCODataset(
        root=ROOT,
        categories_list=class_list,
        train = True,
        clear = False,
        download = False,
        verify = True,
        grid_size = 32,
        return_raw = True,
        anchor_boxes = 5,
    )

    image, label = dataset[25]
    image = np.uint8(image)
    fig, ax = plt.subplots(1, 1)

    print(label)

    ax.imshow(image) 
    ax.set_axis_off() 
    plt.axis('tight') 
    plt.show()

    

