import tensorflow as tf
import numpy as np
import os
import fitz as pdf
import shutil
import cv2
import re
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from pathlib import Path


def trainModel(modelName):
    data = tf.keras.utils.image_dataset_from_directory('data')
    data = data.map(lambda x, y: (x / 255, y))
    train_size = int(len(data) * .7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=7, validation_data=val, callbacks=[tensorboard_callback])

    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())

    model.save(os.path.join('models', modelName))


def evaluateModel(testImagesDir):

    for filename in os.listdir(testImagesDir):
        root = os.path.join(testImagesDir, filename)
        img = cv2.imread(root)
        resize = tf.image.resize(img, (256, 256))
        predictionNew = my_model.predict(np.expand_dims(resize / 255, 0))
        print(filename)
        if predictionNew > 0.5:
            print(f'Predicted class is Keep')
        else:
            print(f'Predicted class is Delete')

def predict_and_delete_pages(pdf_path, output_images_path, output_pdfs_path, model):
    pagesAdded = 0
    doc = pdf.open(pdf_path)
    writer = pdf.open()
    for filename in os.listdir(output_images_path):
        root = os.path.join(output_images_path, filename)

        if not (os.path.isdir(root)):
            keepDir = os.path.join(output_images_path, 'keep')
            keepDir = os.path.join(keepDir, filename)
            deleteDir = os.path.join(output_images_path, 'delete')
            deleteDir = os.path.join(deleteDir, filename)
            img = cv2.imread(root)
            resize = tf.image.resize(img, (256, 256))
            prediction = model.predict(np.expand_dims(resize / 255, 0))
            print(prediction)
            page = (re.search(r'page_\d+', filename))
            page = page.group()
            page = (re.search(r'\d+', page))
            page = int(page.group()) - 1
            print(filename)

            if prediction > 0.7:
                if prediction < 0.90:
                    problems_path = 'Problems'
                    problems_path = os.path.join(problems_path, os.path.basename(pdf_path))
                    writer.insert_pdf(doc)
                    writer.save(problems_path)
                    writer.close()
                if pagesAdded < 2:
                    writer.insert_pdf(doc, from_page=page, to_page=page)
                    pagesAdded += 1
                    Path(root).replace(keepDir)
                    print(f'Predicted class is Keep')
            else:
                Path(root).replace(deleteDir)
                print(f'Predicted class is Delete')

    if pagesAdded > 0:
        writer.save(output_pdfs_path)
        writer.close()
    if pagesAdded == 0 or pagesAdded == 1:
        writer.insert_pdf(doc)
        problems_path = 'Problems'
        problems_path = os.path.join(problems_path, os.path.basename(pdf_path))
        writer.save(problems_path)
        writer.close()
        # shutil.copy(pdf_path, problems_path)

def process_pdfs_in_directory(input_directory, output_directory_images, output_directory_pdfs, model):
    # Iterate through all files in the input directory

    for filename in os.listdir(input_directory):
        if filename.endswith(".pdf"):
            if len(filename) > 40:
                os.rename(os.path.join('Applications', filename), os.path.join('Applications', filename[:40] + '.pdf'))
                filename = filename[:40] + '.pdf'
            os.rename(os.path.join('Applications', filename), os.path.join('Applications', os.path.splitext(filename)[0].strip() + '.pdf'))
            filename = os.path.splitext(filename)[0].strip() + '.pdf'
            pdf_path = os.path.join(input_directory, filename)
            output_images_path = os.path.join(output_directory_images, os.path.splitext(filename)[0])
            output_pdf_path = os.path.join(output_directory_pdfs, filename)
            # Split PDF pages into images

            try:
                split_pdf_pages_to_images(pdf_path, output_images_path)

                # Process images and create a new PDF
                predict_and_delete_pages(pdf_path, output_images_path, output_pdf_path, model)

            except:
                doc = pdf.open(pdf_path)
                writer = pdf.open()
                writer.insert_pdf(doc)
                problems_path = 'Problems'
                problems_path = os.path.join(problems_path, os.path.basename(pdf_path))
                writer.save(problems_path)
                writer.close()


# noinspection PyUnresolvedReferences
def split_pdf_pages_to_images(pdf_path, output_images_path):
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(os.path.join(output_images_path, 'keep'), exist_ok=True)
    os.makedirs(os.path.join(output_images_path, 'delete'), exist_ok=True)
    doc = pdf.open(pdf_path)
    pageNum = 1
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(dpi=150)  # render page to an image
        image_path = os.path.join(output_images_path, f'{os.path.splitext(os.path.basename(pdf_path))[0]} - page_{pageNum}.jpg')
        pix.save(image_path)  # store image as a PNG
        pageNum += 1


def remakeTrainingSets():
    input_directory = 'Images'
    output_directory = 'Temp Training Sets'

    for filename in os.listdir(input_directory):
        studentPath = os.path.join(input_directory, filename)
        if os.path.isdir(studentPath):
            for filename in os.listdir(studentPath):
                directory = os.path.join(studentPath, filename)
                if os.path.isdir(directory):
                    if filename == 'keep':
                        for filename in os.listdir(directory):
                            img = os.path.join(directory, filename)
                            keepPath = os.path.join(output_directory, 'Temp Keep')
                            os.makedirs(keepPath, exist_ok=True)
                            shutil.copy(img, keepPath)
                    else:
                        for filename in os.listdir(directory):
                            img = os.path.join(directory, filename)
                            deletePath = os.path.join(output_directory, 'Temp Delete')
                            os.makedirs(deletePath, exist_ok=True)
                            shutil.copy(img, deletePath)


trainModel('ETS_6.h5')
my_model = load_model('models/ETS_6Ani.h5')


input_directory = 'Applications'
output_directory_images = 'Images'
output_directory_pdfs = 'Trimmed Applications'
testImagesDir = 'New Model Test Images'


# evaluateModel(testImagesDir)
# process_pdfs_in_directory(input_directory, output_directory_images, output_directory_pdfs, my_model)
# remakeTrainingSets()




