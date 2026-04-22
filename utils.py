import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

def gen(pre, train, test):
    train_datagen = ImageDataGenerator(
        preprocessing_function=pre,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    valid_test_datagen = ImageDataGenerator(preprocessing_function=pre)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training'
    )
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0,
        subset='validation'
    )
    test_gen = valid_test_datagen.flow_from_dataframe(
        dataframe=test,
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        verbose=0,
        shuffle=False
    )
    return train_gen, valid_gen, test_gen

def plot_history(history, test_gen, train_gen, model, test_df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model ' + met)
        ax[i].set_ylabel(met)
        ax[i].set_xlabel('Epoch')
        ax[i].legend(['train', 'val'], loc='upper left')

    pred = model.predict(test_gen)
    pred = np.argmax(pred, axis=1)
    labels = {v: k for k, v in train_gen.class_indices.items()}
    pred_labels = [labels[k] for k in pred]

    clr = classification_report(test_df['label'], pred_labels)
    print(clr)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df['filename'].iloc[i + 1]))
        ax.set_title(f'True: {test_df["label"].iloc[i + 1]}, Pred: {pred_labels[i + 1]}')

    plt.tight_layout()
    plt.savefig('figures/history.png', dpi=300)
    plt.savefig('figures/history.pdf', bbox_inches='tight')
    plt.close()
    return history

def result_test(test, model_use):
    results = model_use.evaluate(test, verbose = 0)
    print("Test Loss {:.5f}".format(results[0]))
    print("Test Accuracy {:.2f}".format(results[1] * 100))
    return results
