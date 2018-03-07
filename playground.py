# -*- coding: utf-8 -*-
# @Time    : 2018/3/3 上午12:43
# @Author  : Shark
# @Site    :
# @File    : playground.py
# @Software: PyCharm
import os
import sys

_root = os.path.normpath("%s/../.." % os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
import numpy as np

import pytesseract
import subprocess
import cv2
from PIL import Image, ImageChops
import imutils


def screen_shot():
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    img_np = process.stdout.read()
    img_np = np.fromstring(img_np, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
    # img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
    img_np = cv2.Canny(img_np, 1, 10)
    return img_np


def screen_shot_1():
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    img_np = process.stdout.read()
    img_np = np.fromstring(img_np, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
    img_np = cv2.Canny(img_np, 1, 50)
    return img_np[650:1350, :]


def show(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def find_center(image):
    res1 = cv2.matchTemplate(image, cv2.Canny(cv2.GaussianBlur(cv2.imread('temp_player.jpg', cv2.IMREAD_GRAYSCALE), (5, 5), 0), 1, 10), cv2.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    center1_loc = (max_loc1[0] + 39, max_loc1[1] + 189)
    # 消去小跳棋轮廓对边缘检测结果的干扰
    for k in range(max_loc1[1] - 10, max_loc1[1] + 189):
        for b in range(max_loc1[0] - 10, max_loc1[0] + 100):
            image[k][b] = 0
    cv2.circle(image, center1_loc, 10, 127, 10)

    y_top = np.nonzero(image)[0][0]
    # x 中心点就是这几个像素中心点
    x_center = int(np.mean(np.nonzero(image[y_top])))
    y_bottom = y_top + 100
    for row in range(y_bottom, 1350):
        if image[row, x_center] != 0:
            y_bottom = row
            break
    center = x_center, (y_top + y_bottom) // 2
    cv2.circle(image, center, 10, 127, 10)
    print(np.linalg.norm((np.array(center) - np.array(center1_loc))))
    show(image)

def find_center_better():
    # todo 如果非零位置先变大再变小，可以确认为一个物体
    pass

if __name__ == '__main__':
    image = screen_shot_1()
    find_center(image)

