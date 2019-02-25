#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:10:15 2019

@author: rickers
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input 
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.optimizers import Adam
from DataHandler import DataHandler


class GenerativeNetwork:
    def __init__(self, name):
        self.name = name
        self.img_shape = (32,32,3)
        self.latent_dim = 100

    def build_generator(self):
        def deconv(layer_input,layer_nb,first=False):
            filters = 2**(9-layer_nb)
            if first:
                d = Conv2DTranspose(filters=filters,
                                kernel_size=(2,2),
                                strides=(2,2),
                                padding='valid',
                                name="Deconv_"+str(layer_nb))(layer_input)
            else:
                d = Conv2DTranspose(filters=filters,
                                    kernel_size=(2,2),
                                    strides=(2,2),
                                    padding='same',
                                    name="Deconv_"+str(layer_nb))(layer_input)
            d = BatchNormalization(momentum=0.5)(d)
            d = Activation('relu')(d)
            
            d = Conv2D(filters, (3,3), padding='same',name="Conv_"+str(layer_nb))(d)
            d = BatchNormalization(momentum=0.5)(d)
            d = Activation('relu')(d)
            
            return d
        
        d0 = Input(shape=(self.latent_dim,), name="Input_Gen")
        d1 = Dense(4*4*512,activation='relu')(d0)
        d2 = Reshape((4,4,512))(d1)
        d3 = BatchNormalization()(d2)
        d4 = deconv(d3,1,first=True)
        d5 = deconv(d4,2)
        d6 = deconv(d5,3)
        output = Conv2D(filters=3,
                        kernel_size=(3,3),
                        strides=(1,1),
                        activation='tanh',
                        padding='same',
                        name="Output_Gen")(d6)
        model = Model(d0,output)
        model.summary()
        
        return model
    
    def build_discriminator(self):
        def conv(layer_input,layer_nb,kernel_size,first=False):
            filters = 2**(6+layer_nb)
            
            if first:
                c = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=(2,2),
                       padding='same',
                       name="Conv_"+str(layer_nb))(layer_input)
            else:
                c = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=(2,2),
                           padding='same',
                           name="Conv_"+str(layer_nb))(layer_input)
                c = BatchNormalization(momentum=0.5)(c)
            c = LeakyReLU(alpha=0.2)(c)
            
            return c
        
        c0 = Input(shape=self.img_shape,name="Input_Disc")
        c1 = conv(c0,1,(4,4),first=True)
        c2 = conv(c1,2,(4,4))
        c3 = conv(c2,3,(4,4))
        c4 = Flatten()(c3)
        output = Dense(1,activation='sigmoid',name="Output_Disc")(c4)
        
        model = Model(c0,output)
        model.summary()
        
        return model
        
    def build_gan(self):
        gen = self.build_generator()
        disco = self.build_discriminator()
        
        optimizer = Adam(0.0002,0.5)
        # compile discriminator
        disco.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = gen(z)
        
        # only train generator
        disco.trainable = False
        
        dcgan_output = disco(img)
        
        DCGAN = Model(z,dcgan_output)
        DCGAN.summary()
        
        DCGAN.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        return gen, disco, DCGAN
    
    def save_imgs(self, epoch,batch,generator):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
#        gen_imgs = (gen_imgs + 1) * 127.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])#, cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/Cat_{}_{}.png".format(epoch,batch))
        plt.close()
    
#    def save_imgs(epoch, batch, path, imgs_A, imgs_B, gen_imgs, img_batch, gen=1):
#        path = "images/"
#        r, c = 3, img_batch
#    
#        imgs = np.concatenate([imgs_B, gen_imgs, imgs_A])
#        # Rescale images 0 - 1
#    #    gen_imgs = 0.5 * gen_imgs + 0.5
#    
#        fig, axs = plt.subplots(r, c)
#        titles = ['Vorgabe', 'Generiert', 'Original']
#        cnt = 0
#        
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(imgs[cnt,:,:,0])
#                axs[i,j].axis('off')
#                cnt += 1
#            axs[i, int(img_batch/2)].set_title(titles[i])
#        fig.savefig(path+"gen%d_%d_%d.png" % (gen, epoch, batch))
#        plt.close()
        
    def train(self, epochs,batch_size=128):
        nb_imgs = 5680
        batches = int(5680/128)
        generator, discriminator, DCGAN = self.build_gan()
        
        handler = DataHandler(batch_size=batch_size,nb_imgs=nb_imgs)
        
        for epoch in range(epochs):
            for batch in range(batches):
                # generate noise
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = generator.predict(noise)
                # load batch of cat images
                imgs = handler[batch]

                # Adversarial ground truths
                valid = (1 - 0.9) * np.random.random_sample(batch_size,)+ 0.9
                valid = valid.reshape(batch_size, 1)
                valid = np.round(valid, 2)
                fake = 0.1 * np.random.random_sample(batch_size,)
                fake = fake.reshape(batch_size, 1)
                fake = np.round(fake, 2)
            
                
                # train discriminator
                disco_loss_true = discriminator.train_on_batch(imgs,valid)
                disco_loss_fake = discriminator.train_on_batch(gen_imgs,fake)
                disco_loss = 0.5 * np.add(disco_loss_true, disco_loss_fake)
                
                # train generator
                gen_loss = DCGAN.train_on_batch(noise,valid)
                
                # Plot the progress
                print ("\r[Epoche:{}/{}] [Batch:{}/{}] [D loss: {}, acc.: {}%] [G loss: {}]".format(epoch+1,epochs,
                                                                                    batch,batch_size,
                                                                                    disco_loss[0],
                                                                                    100*disco_loss[1],
                                                                                    gen_loss),
                                                                                    end='')
                if batch % 20 == 0:
                    self.save_imgs(epoch,batch,generator)
                
            # shuffle dataset    
            handler.on_epoch_end()
            
            
         
        
    
    
if __name__ == "__main__":
    gan = GenerativeNetwork("DCGAN")
    gan.train(500)
    