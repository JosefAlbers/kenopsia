from google.colab import drive
drive.mount('/content/gdrive')

# !pip install huggingface_hub

import os
import numpy as np
import cv2 as cv
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

def out_to_matrix(out):
    return np.concatenate((out, [1]), axis=0).reshape(3, 3)

def matrix_to_out(matrix):
    return matrix.reshape(-1)[:-1]

def matrix_to_pts(matrix, size=(224, 224)):
    M = matrix
    orig_pts = [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    orig_pts = np.array(orig_pts, dtype='float32')
    t = np.zeros((4, 3), dtype='float32')
    for i in range(4):
        j = orig_pts[i].reshape(-1, 1)
        t[i] = np.matmul(M, j).reshape(3)
        t[i] = t[i] / t[i, 2]
    t = t[:, :-1]

    t = t + 1
    t = t / 2
    t = t * np.array(size, dtype='float32')

    return t

def perspective_transform(img, matrix, size=(224, 224), fillcolor=None):
    src_pts = np.array([[0, 0],[size[0], 0],[size[0], size[1]],[0, size[1]]], dtype='float32')
    dst_pts = matrix_to_pts(matrix, size=size)
    M = cv.getPerspectiveTransform(dst_pts, src_pts)
    X_SC = img.transform(size, Image.PERSPECTIVE, matrix_to_out(M), Image.BICUBIC, fillcolor=fillcolor)
    return X_SC

def apply(img, matrix, size=None):
    if size is None:
        h, w = img.shape[0], img.shape[1]
    else:
        h, w = size
    pts = matrix_to_pts(matrix, size=(w, h))

    img = perspective_transform(Image.fromarray(img), np.linalg.inv(matrix), fillcolor=0)

    return np.array(img)

def draw_pts(img, matrix):
    pts = matrix_to_pts(matrix)

    new = np.array(img)
    new = cv.drawContours(new, [pts.astype('int')], -1, color=(0, 255, 255), thickness=cv.FILLED)

    for pt in pts:
        new = cv.circle(new, (int(pt[0]), int(pt[1])), 9, (255, 150, 150), -1)
    return new

model_path = 'cp.h5'
hf_hub_download(
    repo_id='Adorg/cp231104',
    filename=model_path,
    repo_type='model',
    local_dir='.'
)
cropper = load_model(model_path)

def go(folder_in='forCrop', folder_out='fromCrop'):
    dir_in = f'/content/gdrive/My Drive/{folder_in}'
    dir_out = f'/content/gdrive/My Drive/{folder_out}'
    images = [(fn, f'{dir_in}/{fn}', f'{dir_out}/{fn}.png') for fn in os.listdir(dir_in)]

    for img_name, path_in, path_out in images:
        img = cv.imread(path_in)
        img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)

        plt.figure()
        plt.subplot(131)
        plt.imshow(img)

        model_in = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        model_in = np.expand_dims(model_in, axis=0)
        model_in = preprocess_input(model_in)
        out = cropper.predict_on_batch(model_in)[0]

        rectified_img = apply(img, out_to_matrix(out))
        cv.imwrite(path_out, rectified_img)

        plt.subplot(132)
        plt.imshow(rectified_img)

        region = np.ones((224, 224, 3), dtype='float32')
        region = draw_pts(region, out_to_matrix(out))
        plt.subplot(133)
        plt.imshow(region)

go()
