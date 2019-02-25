#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:12:32 2019

@author: rickers

DATA STRUCTUR: 
    
|count x height x width x channels|
"""
import os
import sys
import tables
import numpy as np
from PIL import Image
from random import shuffle

class DataHandler:
    def __init__(self, batch_size, nb_imgs):
        self.path = os.getcwd()
        self.data_directory = os.path.join(self.path,os.path.join("CatFace_Dataset_HDF","CatFace_Dataset.hdf"))
        self.batch_size = batch_size
        self.nb_imgs = nb_imgs
        self.index = [i for i in range(self.nb_imgs)]
        self.on_epoch_end()
        
    def __getitem__(self,index):
        idx = self.index[self.batch_size*index:self.batch_size*(index+1)]
        h5file = tables.open_file(self.data_directory, mode='r')
        imgs = h5file.root.Data.Catfaces[idx,:,:,:]
        h5file.close()
        # scale between -1 and 1
        imgs = imgs / 127.5 - 1.
        
        return imgs
    
    def on_epoch_end(self):
        "Shuffles Dataset"
        np.random.shuffle(self.index)
    
        
#    def create_dataset(self):            
#        directory = os.path.join(self.path,"CatFace_Dataset")
#        data_directory = os.path.join(self.path,"CatFace_Dataset_HDF")
#        # create folder for hdf file
#        try:
#            os.makedirs(os.path.join(self.path,"CatFace_Dataset_HDF"), exist_ok=False)
#            print("Data directory: {}".format(data_directory))
#        except Exception as e:
#            print(e,'\n')
#            
#        #data shape
#        shape = (len(os.listdir(directory)),32,32,3)
#        atom = tables.UInt8Atom()
#        # create numpy array of all images
#        data = []
#        for file in os.listdir(directory):
#            im = Image.open(os.path.join(directory,file))
#            data.append(np.array(im))
#        data = np.asarray(data)
#        # create hdf file
#        h5file = tables.open_file(os.path.join(data_directory,"CatFace_Dataset.hdf"), 'w')
#        gcolumns = h5file.create_group(h5file.root, "Data", "Data")
#        h5file.create_carray(gcolumns, 'Catfaces',
#                     atom, shape,
#                     obj=data)
#        h5file.close()
#        
#        stop=1








handler = DataHandler(128,5680)
for i in range(3):
    a = handler[i]