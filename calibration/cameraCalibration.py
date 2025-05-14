# -*- coding: utf-8 -*-
"""
Calibració càmera

Lucía Torrescusa Rubio
Joel Montes de Oca Martínez

"""

import cv2
import numpy as np
import glob


#parámetros tablero
tablero = (7, 7)
medida = 23.0 #milímetros


#matriz del tablero (z=0)
obj = np.zeros((tablero[1]*tablero[0], 3), np.float32)
#coordenadas matriz
obj[:,:2] = np.mgrid[0:tablero[0], 0:tablero[1]].T.reshape(-1,2)
#coordenadas reales
obj *= medida

#puntos reales
objpoint = []
#puntos imagen
imgpoint = []


#importamos imágenes
images = glob.glob('images/*.jpg')


for name in images:
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #busca esquinas
    ret, corners = cv2.findChessboardCorners(gray, tablero, None)
    
    if ret:
        print(f"Esquinas detectadas en {name}")
        objpoint.append(obj)
        #30 iteraciones, 0.001 píxeles precisión mínima (cambiar precisión según calidad)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #ajustar bloques de píxeles según la calidad de la cámara usada (será menor en cámara raspberry)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoint.append(corners2)

        cv2.drawChessboardCorners(img, tablero, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"NO se detectaron esquinas en {name}")

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoint, imgpoint, gray.shape[::-1], None, None)

np.savez("calibration.npz", mtx=mtx, dist=dist)
