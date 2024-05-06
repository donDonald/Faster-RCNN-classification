#!/bin/python3

import os
import sys
import argparse
from Config import Config
from Dataset import Dataset
from Model import Model
from utils import Averager
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image
from xml.etree import ElementTree as et

plt.style.use('ggplot')

def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)




class Classificator:


        def __init__(self, config):
                # Create configuration
                self.config = config

                # Initialize the model and move to the computation device
                model = Model(num_classes=self.config.num_classes)
                self.model = model._model.to(self.config.device)

                # Get the model parameters
                self.params = [p for p in self.model.parameters() if p.requires_grad]

                # Define the optimizer
                self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                # Name to save the trained model with
                #MODEL_NAME = 'model' #PTFIXME - is it needed at all?

                # Setup datasets
                self.train_loader, self.valid_loader = Dataset.setup(self.config)

                # Whether to show transformed images from data loader or not
                if self.config.visualize_transformed_images:
                        from utils import show_tranformed_image
                        show_tranformed_image(self.config.device, self.train_loader)




        def run(self):
                # Initialize the Averager class
                train_loss_hist = Averager()
                val_loss_hist = Averager()
                train_itr = 1
                val_itr = 1

                # Train and validation loss lists to store loss values of all...
                # ... iterations till ena and plot graphs for all iterations
                train_loss_list = []
                val_loss_list = []

                # Start the training epochs
                for epoch in range(self.config.num_epochs):
                        if self.config.verbosity > 0:
                                print(f"\nEPOCH {epoch+1} of {self.config.num_epochs}")

                        # Reset the training and validation loss histories for the current epoch
                        train_loss_hist.reset()
                        val_loss_hist.reset()

                        # Create two subplots, one for each, training and validation
                        figure_1, train_ax = plt.subplots()
                        figure_2, valid_ax = plt.subplots()

                        # Start timer and carry out training and validation
                        start = time.time()
                        train_itr, train_loss = self.train(self.train_loader, train_itr, train_loss_list, train_loss_hist)
                        val_itr, val_loss = self.validate(self.valid_loader, val_itr, val_loss_list, val_loss_hist)

                        end = time.time()
                        if self.config.verbosity > 0:
                                print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
                                print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
                                print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

                        if (epoch+1) % self.config.save_model_epoch == 0: # Save model after every n epochs
                                torch.save(self.model.state_dict(), f"{self.config.out_dir}/model{epoch+1}.pth")
                                if self.config.verbosity > 0:
                                        print('SAVING MODEL COMPLETE...\n')

                        if (epoch+1) % self.config.save_plots_epoch == 0: # Save loss plots after n epochs
                                train_ax.plot(train_loss, color='blue')
                                train_ax.set_xlabel('iterations')
                                train_ax.set_ylabel('train loss')
                                valid_ax.plot(val_loss, color='red')
                                valid_ax.set_xlabel('iterations')
                                valid_ax.set_ylabel('validation loss')
                                figure_1.savefig(f"{self.config.out_dir}/train_loss_{epoch+1}.png")
                                figure_2.savefig(f"{self.config.out_dir}/valid_loss_{epoch+1}.png")
                                if self.config.verbosity > 0:
                                        print('SAVING PLOTS COMPLETE...')

                        if (epoch+1) == self.config.num_epochs: # Save loss plots and model once at the end
                                train_ax.plot(train_loss, color='blue')
                                train_ax.set_xlabel('iterations')
                                train_ax.set_ylabel('train loss')
                                valid_ax.plot(val_loss, color='red')
                                valid_ax.set_xlabel('iterations')
                                valid_ax.set_ylabel('validation loss')
                                figure_1.savefig(f"{self.config.out_dir}/train_loss_{epoch+1}.png")
                                figure_2.savefig(f"{self.config.out_dir}/valid_loss_{epoch+1}.png")
                                torch.save(self.model.state_dict(), f"{self.config.out_dir}/model{epoch+1}.pth")

                        plt.close('all')




        # Running training iterations
        def train(self, loader, itr, loss_list, loss_hist):
                if self.config.verbosity > 0:
                        print('Training')

                # Initialize progress bar
                prog_bar = tqdm(loader, total=len(loader))

                for i, data in enumerate(prog_bar):
                        self.optimizer.zero_grad()
                        images, targets = data

                        images = list(image.to(self.config.device) for image in images)
                        targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item()
                        loss_list.append(loss_value)
                        loss_hist.send(loss_value)
                        losses.backward()
                        self.optimizer.step()
                        itr += 1

                        # Update the loss value beside the progress bar for each iteration
                        prog_bar.set_description(desc=f"Training loss: {loss_value:.4f}")
                return itr, loss_list




        # Running validation iterations
        def validate(self, loader, itr, loss_list, loss_hist):
                if self.config.verbosity > 0:
                        print('Validating')

                # Initialize progress bar
                prog_bar = tqdm(loader, total=len(loader))

                for i, data in enumerate(prog_bar):
                        images, targets = data

                        images = list(image.to(self.config.device) for image in images)
                        targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

                        with torch.no_grad():
                                loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item()
                        loss_list.append(loss_value)
                        loss_hist.send(loss_value)
                        itr += 1

                        # Update the loss value beside the progress bar for each iteration
                        prog_bar.set_description(desc=f"Validation loss: {loss_value:.4f}")
                return itr, loss_list




        # Double-check image Vs annotation
        def checkInputs(self, source):


                def getFiles(source):
                        res = []
                        for root, dirs, files in os.walk(source):
                                for file in files:
                                        if file.endswith('.jpg'):
                                                f = os.path.join(root, file)
                                                res = res + [f]
                        return res

                def getImageResolution(filename):
                        with Image.open(filename) as img:
                                width, height = img.size
                        return width, height


                errors = []
                files = getFiles(source)
                for f in files:
                        w, h = getImageResolution(f)
                        annotation = f + '.xml'
                        tree = et.parse(annotation)
                        root = tree.getroot()
                        size = root.find('size')
                        width = int(size.find('width').text)
                        height = int(size.find('height').text)

                        # Ensure that images size are equal and greater than zero
                        if w != width or h != height or w <= 0 or h <= 0:
                                errors = errors + [f'{f}, image size is incorrect, size:{w, h}, annotation size:{width, height}']

                        # Ensure that bound box is correct
                        bndbox = root.find('object').find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymin = int(bndbox.find('ymin').text)
                        ymax = int(bndbox.find('ymax').text)

                        # Ensure that boundbox sizes ere correct and greater than zero
                        if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0 or xmin >= xmax or ymin >= ymax or\
                           not (xmin <= width and xmax <= width and ymin <= height and ymax <= height):
                                errors = errors + [f'{f}, bound box is incorrect, size:{w, h}, bbox size:{xmin, xmax, ymin, ymax}']

                return errors




if __name__ == '__main__':
        # Create arguments parser
        parser = argparse.ArgumentParser(prog='Classificator',
                                         description='What the program does',
                                         epilog='Text at the bottom of help')

        parser.add_argument('train_dir', help='training images and XML files directory')
        parser.add_argument('valid_dir', help='validation images and XML files directory')
        parser.add_argument('out_dir', help='location to save model and plots')
        parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="set output verbosity level")
        parser.add_argument('-sp', '--save_plots_epoch', type=int, default=1, help='save loss plots after these many epochs')
        parser.add_argument('-sm', '--save_model_epoch', type=int, default=1, help='save model after these many epochs')
        parser.add_argument('-bs', '--batch_size', type=int, default=4, help='increase / decrease according to GPU memeory')
        parser.add_argument('-rt', '--resize_to', type=int, default=512, help='resize the image for training and transforms')
        parser.add_argument('-ne', '--num_epochs', type=int, default=100, help='number of epochs to train for')
        args = parser.parse_args()

        # Create configuration
        config = Config(args)
        #print("#############")
        #print(config)

        # Create engine
        engine = Classificator(config)

        # Check that inputs are correct
        errors = engine.checkInputs(config.train_dir)
        if len(errors):
                eprint(f'{config.train_dir} contains incorrect inputs')
                for e in errors:
                        eprint(e)
                eprint('exiting')
                sys.exit(1)

        errors = engine.checkInputs(config.valid_dir)
        if len(errors):
                eprint(f'{config.valid_dir} contains incorrect inputs')
                for e in errors:
                        eprint(e)
                eprint('exiting')
                sys.exit(1)

        # Do it babe
        engine.run()
