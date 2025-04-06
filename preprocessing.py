import time
import logging
import cv2
import numpy as np
from PIL import ImageOps

def preprocess_image_batch(image_batch):
    """
    Preprocess a batch of images by applying a border, converting to grayscale, and blurring.
    Tries GPU processing and falls back to CPU with identical transformation.
    
    Args:
        image_batch (list): List of PIL.Image objects or numpy arrays.
    
    Returns:
        tuple: (List of processed images, average processing time per image in ms)
    """
    start_time = time.time()
    processed_images = []
    try:
        gpu_frame = cv2.cuda_GpuMat()
        gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (5, 5), 0)
        for img in image_batch:
            img = ImageOps.expand(img, border=10, fill='white')
            img_np = np.array(img) if not isinstance(img, np.ndarray) else img
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gpu_frame.upload(gray)
            gpu_blur = gaussian_filter.apply(gpu_frame)
            processed_images.append(gpu_blur.download())
            del img
    except Exception as e:
        for img in image_batch:
            img = ImageOps.expand(img, border=10, fill='white')
            img_np = np.array(img) if not isinstance(img, np.ndarray) else img
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            processed_images.append(blur)
            del img

    duration = (time.time() - start_time) * 1000 / len(image_batch)
    return processed_images, duration