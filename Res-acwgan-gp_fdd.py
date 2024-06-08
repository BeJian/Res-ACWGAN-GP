# Author: Jian Bi
# Date: 2024/5/13

from numpy import zeros
from numpy import ones
from numpy.random import randn
from sklearn.metrics import *
import keras
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding
from keras.layers import LeakyReLU
import keras.backend as K
from functools import partial
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler  # 0~1
from sklearn.preprocessing import LabelEncoder
import time
import warnings

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.per_process_gpu_memory_fraction = 0.9

latent_dim = 10
select_number = 50  # 5 to 50
class_num = 15
data_dim = 11
num_generate = 100
G_size = select_number * class_num
num_epochs = 20000
threshold = 0.9
datasets = 'SZCAV'
root = 'data_out/SZCAV/ResACWGANGP/50/'

train_hist = dict()
train_hist['D_real_b'] = []
train_hist['D_real_m'] = []
train_hist['D_gp'] = []
train_hist['D_fake_b'] = []
train_hist['D_fake_m'] = []
train_hist['G_b'] = []
train_hist['G_m'] = []
train_hist['F1_score'] = []
train_hist['D_all'] = []
train_hist['G_all'] = []

scaler = MinMaxScaler()
label_encoder = LabelEncoder()


class ACWGANGP():
    def __init__(self):
        self.generator = self.define_generator()
        self.discriminator = self.define_discriminator()

        self.generator.trainable = False
        optimizer = RMSprop(lr=0.0001)

        real_data = tf.keras.Input(shape=(data_dim,))
        real_label = Input(())
        noise = Input(shape=(latent_dim,))
        z_label = Input(())
        fake_data = self.generator([noise, z_label])

        fake, fake_label = self.discriminator(fake_data)
        valid, valid_label = self.discriminator(real_data)

        interpolated_data = self.merge_function([real_data, fake_data])

        validity_interpolated, validity_label = self.discriminator(interpolated_data)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_data)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.D_model = Model(inputs=[real_data, real_label, noise, z_label],
                             outputs=[valid, fake, validity_interpolated, valid_label, fake_label])
        self.D_model.compile(loss=[self.wasserstein_loss,
                                   self.wasserstein_loss,
                                   partial_gp_loss,
                                   'sparse_categorical_crossentropy',
                                   'sparse_categorical_crossentropy'],
                             optimizer=optimizer,
                             loss_weights=[1, 1, 10, 1, 1],
                             metrics=['accuracy'])

        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(latent_dim,))
        z_gen_label = Input(())
        # Generate images based of noise
        gdata = self.generator([z_gen, z_gen_label])
        # Discriminator determines validity
        valid, gen_label = self.discriminator(gdata)
        # Defines generator model
        self.G_model = Model([z_gen, z_gen_label], [valid, gen_label])
        self.G_model.compile(loss=[self.wasserstein_loss,
                                   'sparse_categorical_crossentropy'],
                             optimizer=optimizer)

    def merge_function(self, inputs):
        # alpha = K.random_uniform((32, 1, 1, 1))
        alpha = K.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def define_discriminator(self):
        # weight initialization
        inputs = tf.keras.Input(shape=(data_dim,))
        # inputs = in_shape
        fe = Dense(256)(inputs)
        fe = LeakyReLU()(fe)
        n_hidden = [256]*6
        for hidden in n_hidden:
            ff = fe
            fe = Dense(hidden)(fe)
            fe = LeakyReLU()(fe)
            fe = Dense(hidden)(fe)
            fe = keras.layers.add([fe, ff])
            fe = LeakyReLU()(fe)
        fe = Dropout(0.2)(fe)
        out1 = Dense(1)(fe)
        out2 = Dense(class_num, activation='softmax')(fe)

        return Model(inputs, [out1, out2])

    def define_generator(self):
        latent = Input(shape=(latent_dim,))
        label = Input(())
        # x = keras.layers.concatenate([latent, label], axis=1)
        x = Embedding(10, latent_dim, input_length=1)(label)
        x = keras.layers.concatenate([latent, x])
        fe = Dense(256)(x)
        fe = LeakyReLU()(fe)
        n_hidden = [256]*6
        for hidden in n_hidden:
            ff = fe
            fe = Dense(hidden)(fe)
            fe = LeakyReLU()(fe)
            fe = Dense(hidden)(fe)
            fe = keras.layers.add([fe, ff])
            fe = LeakyReLU()(fe)
        fe = Dropout(0.2)(fe)
        out = Dense(data_dim)(fe)
        out = LeakyReLU()(out)

        return Model([latent, label], out)

    def load_original_data(self):
        data_ = ''
        if datasets == 'SZVAV':
            rdata = pd.read_csv(r"dataset/SZVAV/SZVAV_select_10.csv", sep=',', header='infer')
            norm = pd.read_csv(r"dataset/SZVAV/SZVAV_select_normal_1000.csv", sep=',', header='infer')
            data_ = pd.concat([rdata, norm], ignore_index=True)
        if datasets == 'SZCAV':
            rdata = pd.read_csv(r"dataset/SZCAV/SZCAV_select_"+str(select_number)+".csv", sep=',', header='infer')
            norm = pd.read_csv(r"dataset/SZCAV/SZCAV_select_normal_500.csv", sep=',', header='infer')
            data_ = pd.concat([rdata, norm], ignore_index=True)
        return data_

    def load_select_data(self, data, select_num, save=True):
        # data = pd.read_csv(r"dataset/SZVAV.csv", sep=',', header='infer')
        size = select_num * class_num
        F1 = data[data['fault type'] == 'F1'].sample(n=select_num)
        F2 = data[data['fault type'] == 'F2'].sample(n=select_num)
        F3 = data[data['fault type'] == 'F3'].sample(n=select_num)
        F4 = data[data['fault type'] == 'F4'].sample(n=select_num)
        F5 = data[data['fault type'] == 'F5'].sample(n=select_num)
        F6 = data[data['fault type'] == 'F6'].sample(n=select_num)
        F7 = data[data['fault type'] == 'F7'].sample(n=select_num)
        Normal = data[data['fault type'] == 'Normal'].sample(n=select_num)
        if datasets == 'SZCAV':
            F8 = data[data['fault type'] == 'F8'].sample(n=select_num)
            F9 = data[data['fault type'] == 'F9'].sample(n=select_num)
            F10 = data[data['fault type'] == 'F10'].sample(n=select_num)
            F11 = data[data['fault type'] == 'F11'].sample(n=select_num)
            F12 = data[data['fault type'] == 'F12'].sample(n=select_num)
            F13 = data[data['fault type'] == 'F13'].sample(n=select_num)
            F14 = data[data['fault type'] == 'F14'].sample(n=select_num)
            sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, Normal], ignore_index=True)
        else:
            sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, Normal], ignore_index=True)

        X = sdata.iloc[:, data.columns != "fault type"]
        # X = tsne.fit_transform(X)
        labels = sdata.iloc[:, data.columns == "fault type"].values.reshape(size, 1)
        ytrain_one = label_encoder.fit(labels.ravel())
        labels = ytrain_one.transform(labels)
        y = ones((size, 1)).ravel()
        X_norm = scaler.fit_transform(X)
        # X_norm = np.expand_dims(X_norm.astype(float), axis=2)
        if save:
            np.savetxt(root + 'origin_data/' + "train_data" + str(select_number) + "_level" + "_original.csv", X, delimiter=",")
            np.savetxt(root + 'origin_data/' + "train_data" + str(select_number) + "_level" + "_norm.csv", X_norm, delimiter=",")
            np.savetxt(root + 'origin_data/' + "train_labels" + str(select_number) + ".csv", labels, delimiter=",", fmt='%s')
        return (X_norm, labels), y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        z_ = z_input
        # generate labels
        t = class_num - 1
        fixed_y_ = np.ones((n_samples, 1))
        j = 1
        for i in range(t):
            temp = np.ones((n_samples, 1)) + j
            fixed_y_ = np.concatenate([fixed_y_, temp], 0)
            z_ = np.concatenate([z_, z_input], 0)
            j = j + 1
        label_one = label_encoder.fit(fixed_y_.ravel())
        labels = label_one.transform(fixed_y_)
        return z_, labels

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, latent_dim, n_samples):
        # generate points in latent space
        (z_input, labels_input) = self.generate_latent_points(latent_dim, n_samples)
        data = self.generator.predict((z_input, labels_input))
        # create class labels
        y = zeros((n_samples * class_num, 1))
        return [data, labels_input], y

    # generate samples and save as a plot and save the model
    def summarize_performance(self, d_model):
        rest_data = []
        test_num = 300
        if datasets == 'SZVAV':
            rest_data = pd.read_csv(r"dataset/VAV/SZVAV_test_300.csv", sep=',', header='infer')
        if datasets == 'SZCAV':
            test_num = 150
            rest_data = pd.read_csv(r"dataset/SZCAV/SZCAV_test_150.csv", sep=',', header='infer')

        (X, y), _ = self.load_select_data(rest_data, test_num, save=False)
        # (X, y), _ = self.load_select_chiller_data(rest_data, 100)
        # X = data.iloc[:, data.columns != "fault type"]
        # y = data.iloc[:, data.columns == "fault type"]
        y_one = label_encoder.fit(y.ravel())
        y = y_one.transform(y)
        y_pred, yp = d_model.predict(X)
        # y_pred = y_pred[1:]
        y_pred = np.argmax(yp, axis=1)
        report = classification_report(y, y_pred, digits=4)
        F1 = f1_score(y, y_pred, average='weighted')
        return F1, report

    def show_train_hist(self, hist, show=False, save=False):
        x = range(len(hist['D_all']))
        plt.subplot(1, 2, 1)
        plt.plot(train_hist['D_all'], label='D')
        plt.plot(train_hist['G_b'], label='G')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_hist['F1_score'], label='f1 score')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = root + 'train_hist/'
        if save:
            plt.savefig(path + 'Train_hist.png', dpi=600)
        if show:
            plt.show()
        else:
            plt.close()

    # train the generator and discriminator
    def train(self):
        times = time.time()
        maxF1 = 0
        max_report=[]
        # f1_score_hist = list()
        dataset = self.load_original_data()
        dummy = np.zeros((G_size, 1))
        d_real = -ones((G_size, 1))  # 为假样本创建反向标签
        d_fake = ones((G_size, 1))  # 为假样本创建反向标签
        dg_real = -ones((class_num*num_generate, 1))  # 为假样本创建反向标签
        zreal = np.random.normal(0, 1, (G_size, latent_dim))

        for i in range(num_epochs):  # 循环每个训练周期
            (X_real, labels_real), y_real = self.load_select_data(dataset, select_number, save=False)
            d_loss = self.D_model.train_on_batch([X_real, labels_real, zreal, labels_real],
                                                 [d_real, d_fake, dummy, labels_real, labels_real])
            train_hist['D_all'].append(d_loss[0])
            train_hist['D_real_b'].append(d_loss[1])
            train_hist['D_fake_b'].append(d_loss[2])
            train_hist['D_gp'].append(d_loss[3])
            train_hist['D_real_m'].append(d_loss[4])
            train_hist['D_fake_m'].append(d_loss[5])

            (z_input, labels_input) = self.generate_latent_points(latent_dim, num_generate)
            g_loss = self.G_model.train_on_batch([z_input, labels_input],
                                                 [dg_real, labels_input])
            train_hist['G_all'].append(g_loss[0])
            train_hist['G_b'].append(g_loss[1])
            train_hist['G_m'].append(g_loss[2])

            F1_score, report = self.summarize_performance(self.discriminator)
            train_hist['F1_score'].append(F1_score)
            if F1_score > maxF1:
                maxF1 = F1_score
                max_report = report
                if maxF1 > threshold:
                    # save model
                    filename2 = root + 'model/G/g_model_%.4f.h5' % maxF1
                    self.generator.save(filename2)
                    filename3 = root + 'model/D/d_model_%.4f.h5' % maxF1
                    self.discriminator.save(filename3)

                    X_fake = self.generator.predict([z_input, labels_input])
                    X_fake = scaler.inverse_transform(X_fake)
                    X_fake_temp = X_fake
                    labels_fake_temp = labels_input
                    X_fake_temp = X_fake_temp.reshape(num_generate * class_num, data_dim)
                    labels_fake_temp = labels_fake_temp.reshape(num_generate * class_num, 1)
                    generated_fake_data = np.append(X_fake_temp, labels_fake_temp, axis=1)
                    filename4 = root + 'generated_data/generated_fake_data_%.4f.csv' % maxF1
                    np.savetxt(filename4, generated_fake_data, delimiter=",", fmt='%.2f')

                    print('--Saved: %s, %s and %s' % (filename2, filename3, filename4))

            print("%d [Dr loss: %.2f] [Df loss: %.2f] [Dgp loss: %.2f] [Dr Closs: %.2f] [Df Closs: %.2f] [G loss: %.2f] [G Closs: %.2f] [F1: %.4f]" %
                  (i, d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5], g_loss[1], g_loss[2], F1_score))

        # plot WGAN history
        print("training time: %.4f s" % (time.time() - times))
        print('max_f1: %.4f' % max(train_hist['F1_score']), ' index: %d' % np.where(train_hist['F1_score'] == max(train_hist['F1_score'])))
        print(max_report)
        self.show_train_hist(train_hist, show=True, save=True)


if __name__ == '__main__':
    if not os.path.isdir(root + 'origin_data'):
        os.mkdir(root + 'origin_data')
    if not os.path.isdir(root + 'generated_data'):
        os.mkdir(root + 'generated_data')
    if not os.path.isdir(root + 'train_hist'):
        os.mkdir(root + 'train_hist')
    if not os.path.isdir(root + 'model'):
        os.mkdir(root + 'model')
    if not os.path.isdir(root + 'model/' + 'D'):
        os.mkdir(root + 'model/' + 'D')
    if not os.path.isdir(root + 'model/' + 'G'):
        os.mkdir(root + 'model/' + 'G')
    acwgan = ACWGANGP()
    acwgan.train()
