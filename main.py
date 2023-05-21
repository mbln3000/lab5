# 1.	Работа с изображениями:
# a)	Перевод в градации серого и в чёрно-белое изображение по порогу

import cv2
from matplotlib import pyplot as plt

# image = cv2.imread("img.png")
# # grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # black and white
# (thresh, bnw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# cv2.imshow('Original image',image)
# cv2.imshow('Gray image', gray)
# cv2.imshow('Black white image', bnw)
#
# # b)	Добавление надписей, кадрирование, изменение размера
#
# # adding text
# window_name = 'Image with Text'
# font = cv2.FONT_HERSHEY_SIMPLEX
# org = (50, 50)
# fontScale = 1
# # Blue color in BGR
# color = (255, 0, 0)
# thickness = 2
# imagetext = cv2.putText(image, 'Text', org, font,
#                     fontScale, color, thickness, cv2.LINE_AA)
#
# cv2.imshow(window_name, imagetext)
#
# #crop an image
# cropped_image = image[0:700, 300:1200]
# cv2.imshow("Cropped", cropped_image)
#
# # change resolution
# scale_percent = 60
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
# cv2.imshow("Resized image", resized)
#
# # c)	Поворот изображения, размытие и сглаживание.
#
# # rotate
# window_name = 'Rotated Image'
# rot_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#
# cv2.imshow(window_name, rot_image)
#
# # blur
# window_name = 'Blurred Image'
# ksize = (5, 5)
# bl_image = cv2.blur(image, ksize)
# cv2.imshow(window_name, bl_image)
#
# # smoothing
# window_name = 'Smooth Image'
# smooth = cv2.bilateralFilter(image,9,50,50)
# cv2.imshow(window_name, smooth)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2. Распознавание лиц на фото. Написать алгоритм,
# который будет распознавать лица на фотографии,
# обводить их прямоугольными рамками и выводить количество найденных лиц.

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# img = cv2.imread('faces.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cv2.imshow('Faces', img)
# cv2.waitKey()

# 3. Рисование. С помощью opencv python написать функции, которые по действиям мыши
# будут рисовать фигуры или переключаться в режим свободного рисования. Необходимо:
# i.	Создать пустое окно с чёрным фоном
# ii.	Реализовать функцию построения фигуры при двойном нажатии левой кнопки мыши
# окружность
# iii.	Реализовать функцию свободного рисования при нажатии и удерживании правой кнопки
# iv.	*Дополнительные функции, если потребуются
# v.	Добавить подпись
# vi.	Реализовать прерывание программы и закрытие окна при нажатии клавиши ‘q’

import numpy as np


drawing = False
pt1_x, pt1_y = None, None


def line_drawing(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        drawing=False
        cv2.line(img, (pt1_x, pt1_y), (x, y),color=(255, 255, 255), thickness=3)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('test draw')

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
imagetext = cv2.putText(img, 'Hooray!!', org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

cv2.setMouseCallback('test draw', line_drawing)

while (1):
    cv2.imshow('test draw', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

k = cv2.waitKey(0) & 0xFF
if k == 27:  # закрывается клавишей esc
    cv2.destroyAllWindows()


# 3. Обнаружение движущихся объектов на видео. Задачи:
# a)	Окрыть видео в opencv
# b)	Использовать алгоритм вычитания фона (fgMask = bgSubtractor.apply)
# и морфологические операции для удаления шума
# c)	Настроить границы (контуры) прямоугольников или иных фигур вокруг движущихся объектов
# d)	Отобразить результат.
import argparse
from get_background import get_background


def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

get_background('video.mp4')