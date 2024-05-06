#!/bin/python3

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import glob

from xml.etree import ElementTree as et
from torch.utils.data import Dataset as BasicDataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform
from Config import Config




# Dataset class
class Dataset(BasicDataset):


        @staticmethod
        def setup(config: Config):
                # prepare the final datasets and data loaders
                train_dataset = Dataset(
                        config.train_dir,
                        config.resize_to,
                        config.resize_to,
                        config.classes,
                        get_train_transform())

                valid_dataset = Dataset(
                        config.valid_dir,
                        config.resize_to,
                        config.resize_to,
                        config.classes,
                        get_valid_transform())

                train_loader = DataLoader(
                        train_dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate_fn
                )

                valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn
                )
                if config.verbosity > 1:
                        print(f"Number of training samples: {len(train_dataset)}")
                        print(f"Number of validation samples: {len(valid_dataset)}\n")
                return train_loader, valid_loader


        #def __init__(self, config, dir_path, width, height, classes, transforms=None):
        def __init__(self, dir_path, width, height, classes, transforms=None):
                #self.config = config
                self.transforms = transforms
                self.dir_path = dir_path
                self.height = height
                self.width = width
                self.classes = classes

                # get all the image paths in sorted order
                self.image_paths = glob.glob(f"{self.dir_path}/**/*.jpg")
                self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
                self.all_images = sorted(self.all_images)
               #print(f'data set for "{dir_path}":')
               #for i in self.all_images:
               #        print(f'    {i}')


        def __getitem__(self, idx):
                # Collect image name and full image path
                image_name = self.image_paths[idx]
                image_path = image_name
                # Read the image
                image = cv2.imread(image_path)
                # Convert BGR to RGB color format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image_resized = cv2.resize(image, (self.width, self.height))
                image_resized /= 255.0

                # Collect corresponding annotation XML file
                annot_filename = image_name + '.xml'
                annot_file_path = annot_filename

                boxes = []
                labels = []
                tree = et.parse(annot_file_path)
                root = tree.getroot()

                # get the height and width of the image
                image_width = image.shape[1]
                image_height = image.shape[0]

                # box coordinates for xml files are extracted and corrected for image size given
                for member in root.findall('object'):
                        # map the current object name to `classes` list to get...
                        # ... the label index and append to `labels` list
                        labels.append(self.classes.index(member.find('name').text))

                        # xmin = left corner x-coordinates
                        xmin = int(member.find('bndbox').find('xmin').text)
                        # xmax = right corner x-coordinates
                        xmax = int(member.find('bndbox').find('xmax').text)
                        # ymin = left corner y-coordinates
                        ymin = int(member.find('bndbox').find('ymin').text)
                        # ymax = right corner y-coordinates
                        ymax = int(member.find('bndbox').find('ymax').text)

                        # resize the bounding boxes according to desired `width`, `height`
                        #PTFIXME, review these resizing
                        xmin_final = (xmin/image_width)*self.width
                        xmax_final = (xmax/image_width)*self.width
                        ymin_final = (ymin/image_height)*self.height
                        yamx_final = (ymax/image_height)*self.height

                        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

                # bounding box to tensor
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # area of the bounding boxes
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # no crowd instances
                iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                # labels to tensor
                labels = torch.as_tensor(labels, dtype=torch.int64)

                # prepare the final `target` dictionary
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
                image_id = torch.tensor([idx])
                target["image_id"] = image_id

                # apply the image transforms
                if self.transforms:
                        sample = self.transforms(image = image_resized,
                                                 bboxes = target['boxes'],
                                                 labels = labels)
                        image_resized = sample['image']
                        target['boxes'] = torch.Tensor(sample['bboxes'])

                return image_resized, target


        def __len__(self):
                return len(self.image_paths)




# Execute Dataset.py using Python command from Terminal to visualize sample images
# USAGE: Dataset.py some_samples_folder
if __name__ == '__main__':
        parser = argparse.ArgumentParser(prog='Dataset',
                                         description='Create a dataset for current folder and print images one by one')

        parser.add_argument('input', help='images and XML files directory')
        parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="set output verbosity level")
        parser.add_argument('-bs', '--batch_size', type=int, default=4, help='increase / decrease according to GPU memeory')
        parser.add_argument('-rt', '--resize_to', type=int, default=512, help='resize the image for training and transforms')
        args = parser.parse_args()

        classes = next(os.walk(args.input))[1]
        dataset = Dataset(
                args.input,
                args.resize_to,
                args.resize_to,
                classes
        )

        def visualize_sample(image, target):
                box = target['boxes'][0]
                label = classes[target['labels']]
                cv2.rectangle(
                        image,
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        (0, 0, 255), 1
                )
                cv2.putText(
                        image, label, (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                cv2.imshow('Image', image)
                return cv2.waitKey(0)

        index=0
        while True:
                image, target = dataset[index]
                key = visualize_sample(image, target)
                match key:
                        case 27: sys.exit(0) # Escape
                        case 113: sys.exit(0) # Q
                        case 83: index = (index + 1) % len(dataset) # Left
                        case 84: index = (index + 1) % len(dataset) # Down
                        case 81: index = index - 1 if index > 0 else len(dataset)-1 # Right
                        case 82: index = index - 1 if index > 0 else len(dataset)-1 # Up
                #print(f'index:{index}, key:{key}')
