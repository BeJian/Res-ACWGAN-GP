from numpy import zeros
from numpy import ones
from numpy.random import randn
from sklearn.metrics import *
import keras
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding
from keras.layers import LeakyReLU
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler  # 0~1
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.per_process_gpu_memory_fraction = 0.9

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(tf.test.is_built_with_cuda())
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.per_process_gpu_memory_fraction = 0.9
warnings.filterwarnings('ignore')

latent_dim = 5
select_number = 50
class_num = 8
data_dim = 12
num_epochs = 20000
num_generate = 100
thresh = 0.9
datasets = "SZVAV"
root = 'data_out/SZVAV/ACGAN/50/'

train_hist = dict()
train_hist['D_real_b'] = []
train_hist['D_real_m'] = []
train_hist['D_fake_b'] = []
train_hist['D_fake_m'] = []
train_hist['G_b'] = []
train_hist['G_m'] = []
train_hist['F1_score'] = []
train_hist['D_all'] = []
train_hist['G_all'] = []

scaler = MinMaxScaler()
label_encoder = LabelEncoder()


def define_discriminator():
    # weight initialization
    inputs = tf.keras.Input(shape=(data_dim,))
    # inputs = in_shape
    fe = Dense(256)(inputs)
    fe = LeakyReLU()(fe)
    n_hidden = [256]*10
    for hidden in n_hidden:
        fe = Dense(hidden)(fe)
        fe = LeakyReLU()(fe)

    fe = Dropout(0.2)(fe)
    out1 = Dense(1, activation='sigmoid')(fe)
    out2 = Dense(class_num, activation='softmax')(fe)
    model = Model(inputs, [out1, out2])
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
                  , optimizer=opt
                  )

    return model


def define_generator():
    latent = Input(shape=(latent_dim,))
    label = Input(shape=())
    x = Embedding(class_num, latent_dim, input_length=1)(label)
    x = keras.layers.concatenate([latent, x], 1)
    fe = Dense(256)(x)
    fe = LeakyReLU()(fe)
    n_hidden = [256]*8
    for hidden in n_hidden:
        fe = Dense(hidden)(fe)
        fe = LeakyReLU()(fe)

    gen = Dropout(0.2)(fe)
    out = Dense(data_dim)(gen)
    out = LeakyReLU()(out)
    model = Model([latent, label], out)
    model.summary()
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy'
                        ,'sparse_categorical_crossentropy'],
                  optimizer=opt
                  )
    return model


def load_original_data(D_name):
    data_ = ''
    if D_name == 'SZVAV':
        file = "dataset/SZVAV/SZVAV_select_" + str(select_number) + ".csv"
        rdata = pd.read_csv(file, sep=',', header='infer')
        norm = pd.read_csv("dataset/SZVAV/SZVAV_select_normal_1000.csv", sep=',', header='infer')
        data_ = pd.concat([rdata, norm], ignore_index=True)
    if D_name == 'SZCAV':
        file = "dataset/SZCAV/SZCAV_select_" + str(select_number) + ".csv"
        rdata = pd.read_csv(file, sep=',', header='infer')
        norm = pd.read_csv("dataset/SZCAV/SZCAV_select_normal_500.csv", sep=',', header='infer')
        data_ = pd.concat([rdata, norm], ignore_index=True)
    return data_


def load_select_data(data, select_num, save=True):
    # data = pd.read_csv(r"dataset/SZVAV_select.csv", sep=',', header='infer')
    size = select_num * class_num
    F1 = data[data['fault type'] == 'F1'].sample(n=select_num)
    F2 = data[data['fault type'] == 'F2'].sample(n=select_num)
    F3 = data[data['fault type'] == 'F3'].sample(n=select_num)
    F4 = data[data['fault type'] == 'F4'].sample(n=select_num)
    F5 = data[data['fault type'] == 'F5'].sample(n=select_num)
    F6 = data[data['fault type'] == 'F6'].sample(n=select_num)
    F7 = data[data['fault type'] == 'F7'].sample(n=select_num)
    Normal = data[data['fault type'] == 'Normal'].sample(n=select_num)
    if datasets=='SZCAV':
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
    labels = sdata.iloc[:, data.columns == "fault type"].values.reshape(size, 1)
    ytrain_one = label_encoder.fit(labels.ravel())
    labels = ytrain_one.transform(labels)
    y = ones((size, 1)).ravel()
    X_norm = scaler.fit_transform(X)
    # X_norm = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))
    # X_norm = np.expand_dims(X_norm.astype(float), axis=2)
    if save:
        np.savetxt(root + 'origin_data/' + "train_data" + str(select_number) + "_original.csv", X, delimiter=",", fmt='%.2f')
        np.savetxt(root + 'origin_data/' + "train_data" + str(select_number) + "_norm.csv", X_norm, delimiter=",", fmt='%.2f')
        np.savetxt(root + 'origin_data/' + "train_labels" + str(select_number) + ".csv", labels, delimiter=",", fmt='%s')
    return (X_norm, labels), y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=class_num):
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
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    (z_input, labels_input) = generate_latent_points(latent_dim, n_samples)
    data = generator.predict((z_input, labels_input))
    # create class labels
    y = zeros((n_samples * class_num, 1))
    return [data, labels_input], y


# generate samples and save as a plot and save the model
def summarize_performance(d_model):
    if datasets == 'AHU':
        rest_data = pd.read_csv(r"dataset/AHU/AHU_test_300.csv", sep=',', header='infer')
    if datasets == 'SZVAV':
        rest_data = pd.read_csv(r"dataset/SZVAV/SZVAV_test_300.csv", sep=',', header='infer')
    if datasets == 'SZCAV':
        rest_data = pd.read_csv(r"dataset/SZCAV/SZCAV_test_150.csv", sep=',', header='infer')
    (X, y), _ = load_select_data(rest_data, 300, save=False)
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


def show_train_hist(hist, show=False, save=False):
    x = range(len(hist['D_all']))
    plt.subplot(1, 2, 1)
    plt.plot(hist['D_all'], label='D')
    plt.plot(hist['G_all'], label='G')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist['F1_score'], label='F1 score')
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
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=num_epochs, n_generate=num_generate):
    times = time.time()
    maxF1 = 0
    max_report = []
    for i in range(n_epochs):
        (X_real, labels_real), y_real = load_select_data(dataset, select_number)

        d_real_loss, d_r1, d_r2 = d_model.train_on_batch(X_real, (y_real, labels_real))

        (X_fake, labels_fake), y_fake = generate_fake_samples(g_model, latent_dim, n_generate)
        d_fake_loss, d_f1, d_f2 = d_model.train_on_batch(X_fake, (y_fake, labels_fake))

        D_loss = (d_real_loss + d_fake_loss) * 0.5

        (z_input, z_labels) = generate_latent_points(latent_dim, n_generate)
        y_gan = ones((n_generate * class_num, 1))

        G_loss, g_1, g_2 = gan_model.train_on_batch((z_input, z_labels), (y_gan, z_labels))

        D_F1_score, reports = summarize_performance(d_model)
        train_hist['F1_score'].append(np.mean(D_F1_score))
        print('>%d, [D_loss: %.3f], [G_loss: %.3f], [test_f1_score: %.4f]'
              % (i + 1, D_loss, G_loss, D_F1_score))

        train_hist['D_real_b'].append(np.mean(d_r1))
        train_hist['D_real_m'].append(np.mean(d_r2))
        train_hist['D_fake_b'].append(np.mean(d_f1))
        train_hist['D_fake_m'].append(np.mean(d_f2))
        train_hist['G_b'].append(np.mean(g_1))
        train_hist['G_m'].append(np.mean(g_2))
        train_hist['D_all'].append(np.mean(D_loss))
        train_hist['G_all'].append(np.mean(G_loss))

        if D_F1_score > maxF1:
            maxF1 = D_F1_score
            max_report = reports
            if maxF1 > thresh:
                # save model
                filename2 = root + 'model/G/g_model_%.4f.h5' % maxF1
                g_model.save(filename2)
                filename3 = root + 'model/D/d_model_%.4f.h5' % maxF1
                d_model.save(filename3)
                filename4 = root + 'model/GAN/gan_model_%.4f.h5' % maxF1
                gan_model.save(filename4)
                print('--Saved: %s, %s and %s' % (filename2, filename3, filename4))

                X_fake = scaler.inverse_transform(X_fake)
                X_fake_temp = X_fake
                labels_fake_temp = labels_fake
                X_fake_temp = X_fake_temp.reshape(n_generate * class_num, data_dim)
                labels_fake_temp = labels_fake_temp.reshape(n_generate * class_num, 1)
                generated_fake_data = np.append(X_fake_temp, labels_fake_temp, axis=1)
                np.savetxt(root + 'generated_data/generated_fake_data_%.4f.csv' % maxF1, generated_fake_data, delimiter=",", fmt='%.2f')

    F1_max = max(train_hist['F1_score'])
    print('max_f1: %.4f' % F1_max, ' index: %d' % np.where(train_hist['F1_score'] == F1_max))
    print(max_report)
    print("training time: %.4f s" % (time.time() - times))
    show_train_hist(train_hist, show=True, save=True)


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

discriminator = define_discriminator()
generator = define_generator()
gan_model = define_gan(generator, discriminator)
dataset = load_original_data(datasets)
train(generator, discriminator, gan_model, dataset, latent_dim)
