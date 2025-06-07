from flask import Flask, render_template, Response
import cv2
import numpy as np
from io import BytesIO

# sketch effect 
def sketch(frame):
    # mengubah gambar ke grayscale terlebih dahulu
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # membersihkan noise dengan Gaussian blur
    frame_blur= cv2.GaussianBlur(frame_gray,(5,5),0)
    # emplementasi deteksi tepi dengan Canny
    edges = cv2.Canny(frame_blur,30,60)
    # do an inverse binary thresholding
    ret, mask = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY_INV)

    return mask