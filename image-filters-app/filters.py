import cv2
import numpy as np
import streamlit as st


@st.cache
def bw_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


@st.cache
def vignette(img, level=2):
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

    # Generating resultant_kernel matrix
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Apply the mask to each channel in the input image
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette


@st.cache
def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255]
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia


@st.cache
def pencil_sketch(img, ksize=5):
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)  # noqa
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch


@st.cache
def oil_painting(img):
    return cv2.xphoto.oilPainting(img, 7, 1)


@st.cache
def invert(img):
    return cv2.bitwise_not(img)
