# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:34:10 2023

@author: julayar.1
"""

import mediapipe as mp
import cv2 as cv
import numpy as np

# %% Calibración

from FuncionesCalibraciónComentadas import CalibracionIndividual
from FuncionesCalibraciónComentadas import CalibracionEstereo
from FuncionesCalibraciónComentadas import VinculaMatrices
from FuncionesCalibraciónComentadas import Triangulacion
from FuncionesCalibraciónComentadas import AbrirVideo
from FuncionesCalibraciónComentadas import PredecirFrame

ficheroImagenes1 = "Calibracion/Camara1/*"
ficheroImagenes2 = "Calibracion/Camara2/*"

filas = 4
columnas = 7
tamano = 0.024

Kmatriz1, dist1 = CalibracionIndividual(ficheroImagenes1, filas, columnas, tamano)
Kmatriz2, dist2 = CalibracionIndividual(ficheroImagenes2, filas, columnas, tamano)

Rmatriz, Tvector = CalibracionEstereo(ficheroImagenes1,
                                      ficheroImagenes2,
                                      filas,
                                      columnas,
                                      tamano,
                                      Kmatriz1,
                                      dist1,
                                      Kmatriz2,
                                      dist2)

Pmatriz1, Pmatriz2 = VinculaMatrices(Rmatriz, Tvector, Kmatriz1, Kmatriz2)

# %% Abrir videos

videoDerecha = 'Videos/Der90.avi'
videoIzquierda = 'Videos/Izq90.avi'

camDer, NframesDer, fpsDer = AbrirVideo(videoDerecha)
camIzq, NframesIzq, fpsIzq = AbrirVideo(videoIzquierda)

if NframesDer != NframesIzq:
    print("Error: videos de distinto tamaño.")
    print("Nº frames video derecho: ", NframesDer,"\n","Nº frames video izquierdo:" , NframesIzq)
    
# %% Extraer coordenadas

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

coordenadasDerecha = []
coordenadasIzquierda = []

hands = mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 1)
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 1) as hands:
    for frame in range(NframesDer):
        resultadosDer, imagenDer = PredecirFrame(camDer, hands)
        resultadosIzq, imagenIzq = PredecirFrame(camIzq, hands)
        
        if resultadosDer.multi_hand_landmarks and resultadosIzq.multi_hand_landmarks:
            mp_drawing.draw_landmarks(imagenDer, resultadosDer.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(imagenIzq, resultadosIzq.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cooDer = np.zeros((21*3))
            for coord in np.asarray(range(21)):
                cooDer[3*coord:3*coord+3] =  np.array([resultadosDer.multi_hand_landmarks[0].landmark[coord].x, resultadosDer.multi_hand_landmarks[0].landmark[coord].y, resultadosDer.multi_hand_landmarks[0].landmark[coord].z*(-1)])
            coordenadasDerecha.append(cooDer)
            cooIzq = np.zeros((21*3))
            for coord in np.asarray(range(21)):
                cooIzq[3*coord:3*coord+3] =  np.array([resultadosIzq.multi_hand_landmarks[0].landmark[coord].x, resultadosIzq.multi_hand_landmarks[0].landmark[coord].y, resultadosIzq.multi_hand_landmarks[0].landmark[coord].z*(-1)])
            coordenadasIzquierda.append(cooIzq)
        #cv.imshow(videoDer, imagenDer)
        #cv.imshow(videoIzq, imagenIzq)
        cv.waitKey(int(1/fpsDer*1000))
        print(frame+1,"/", NframesDer)
    #cv.destroyWindow(videoDer)
    #cv.destroyWindow(videoIzq)
    
coordenadasDer = np.array(coordenadasDerecha)
coordenadasIzq = np.array(coordenadasIzquierda)

rows, cols = coordenadasDer.shape

# %% Triangulación

Puntos3D = np.empty((rows,0))     
        
for i in range(0,cols, 3):
    print(int(i/3)+1,"/", int(cols/3))
    PuntosDer = np.multiply(coordenadasDer[:,[i,i+1]].astype('float64'),[800, 480])
    PuntosIzq = np.multiply(coordenadasIzq[:,[i,i+1]].astype('float64'),[800, 480])
    Coord3D = Triangulacion(PuntosDer, PuntosIzq, Pmatriz1, Pmatriz2)
    Puntos3D = np.concatenate((Puntos3D, Coord3D.reshape(-1,3)), axis = 1)
    
t = np.arange(0.0, NframesDer/fpsDer, (NframesDer/fpsDer)/len(Puntos3D))
