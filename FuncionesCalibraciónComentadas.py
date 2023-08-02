# -*- coding: utf-8 -*-

import glob
import cv2 as cv
import numpy as np
from scipy import linalg



def CalibracionIndividual(fichero_imagenes, filas, columnas, tamano):
    """
    Toma todas las fotos dentro del fichero seleccionado y realiza un calibrado
    de la cámara con la que se han tomado especificando las características del
    patrón de ajedrez como variables de entrada.                                                                              
    """
    
    # Crear listado de fotos
    rutas = sorted(glob.glob(fichero_imagenes))
    imagenes = []
    
    for imagen in rutas[0:len(rutas)-1]:
        im = cv.imread(imagen,1)
        imagenes.append(im)
    
    # Definir criterios de terminación del algoritmo iterativo calibrateCamera
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    # Describir el patrón en coordenadas coplanares medidas en metros
    patron = np.zeros((filas*columnas, 3), np.float32)
    patron[:,:2] = np.mgrid[0:filas, 0:columnas].T.reshape(-1,2)
    patron = tamano * patron
    
    ancho = imagenes[0].shape[1]
    alto = imagenes[0].shape[0]
    
    puntos_imagen = []
    
    puntos_calibracion = []
    
    # Recorrer cada foto y detectar las esquinas interiores del patrón
    for imagen in imagenes:
        BW = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        ret, esquinas = cv.findChessboardCorners(BW, (filas, columnas), None)
        
        # Si se encuentra el patrón en la foto, ret == True
        if ret == True:
            
            # Tamaño de convolución para detectar las esquinas. Si es muy grande produce errores.
            convolucion = (5, 5)
            
            # Refinar la detección de esquinas y mostrarlas superpuestas en la foto
            esquinas = cv.cornerSubPix(BW, esquinas, convolucion, (-1, -1), criteria)
            cv.drawChessboardCorners(imagen, (filas, columnas), esquinas, ret)
            cv.imshow('Patron detectado', imagen)
            cv.waitKey()
            
            # Guardar las coordenadas obtenidas
            puntos_calibracion.append(patron)
            puntos_imagen.append(esquinas)
    
    # Calibración propiamente dicha. Estima la mejor matriz de calibración
    # asociada a las fotos y la distorsión de la cámara. También devuelve las
    # matrices de rotación y los vectores de traslación asociadas a cada foto.
    ret, matrizK, distorsion, rotacion, traslacion = cv.calibrateCamera(puntos_calibracion,
                                                                        puntos_imagen,
                                                                        (ancho, alto),
                                                                        None,
                                                                        None)
    return matrizK, distorsion

def CalibracionEstereo(fichero_imagenes1,
                       fichero_imagenes2,
                       filas, columnas,
                       tamano,
                       Kmatriz1,
                       distorsion1,
                       Kmatriz2,
                       distorsion2):
    """
    Toma dos conjuntos de imágenes, correspondientes a dos cámaras diferentes,
    de los ficheros especificados. Realiza con ellos una calibración estéreo,
    tomando como variables de entrada las características del patrón usado.
    """
    
    # Crear listados de fotos
    rutas1 = sorted(glob.glob(fichero_imagenes1))
    rutas2 = sorted(glob.glob(fichero_imagenes2))
    
    imagenes1 = []
    imagenes2 = []
    
    for imagen1, imagen2 in zip(rutas1[0:len(rutas1)-1], rutas2[0:len(rutas2)-1]):
        aux = cv.imread(imagen1, 1)
        imagenes1.append(aux)
        
        aux = cv.imread(imagen2, 1)
        imagenes2.append(aux)
     
    # Definir criterios de terminación del algoritmo iterativo calibrateCamera
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        
    # Describir el patrón en coordenadas coplanares medidas en metros
    patron = np.zeros((filas*columnas, 3), np.float32)
    patron[:,:2] = np.mgrid[0:filas, 0:columnas].T.reshape(-1,2)
    patron = tamano * patron
    
    ancho = imagenes1[0].shape[1]
    alto = imagenes1[0].shape[0]
    
    puntos_imagenes1 = []
    puntos_imagenes2 = []
    
    puntos_calibracion = []
    
    
    # Recorrer cada pareja de fotos y detectar las esquinas interiores del patrón
    for imagen1, imagen2 in zip (imagenes1, imagenes2):
        BW1 = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
        BW2 = cv.cvtColor(imagen2, cv.COLOR_BGR2GRAY)
        ret1, esquinas1 = cv.findChessboardCorners(BW1, (filas, columnas), None)
        ret2, esquinas2 = cv.findChessboardCorners(BW2, (filas, columnas), None)
        

        
        # Si se encuentra el patrón en ambas fotos, ret1 == ret2 == True
        if ret1 == True and ret2 == True:
            
            # Tamaño de convolución para detectar las esquinas. Si es muy grande produce errores.
            convolucion = (5, 5)
            
            # Refinar la detección de esquinas y mostrarlas superpuestas en las fotos
            esquinas1 = cv.cornerSubPix(BW1, esquinas1, convolucion, (-1, -1), criteria)
            esquinas2 = cv.cornerSubPix(BW2, esquinas2, convolucion, (-1, -1), criteria)
            
            cv.drawChessboardCorners(imagen1, (filas, columnas), esquinas1, ret1)
            cv.imshow('Patron detectado 1', imagen1)
            
            cv.drawChessboardCorners(imagen2, (filas, columnas), esquinas2, ret2)
            cv.imshow('Patron detectado 2', imagen2)
            
            # Guardar las coordenadas obtenidas
            puntos_calibracion.append(patron)
            puntos_imagenes1.append(esquinas1)
            puntos_imagenes2.append(esquinas2)
            cv.waitKey()
    
    # Calibración estéreo propiamente dicha. Estima la mejor matriz de rotación
    # R y el mejor vector de traslación T entre las dos cámaras que mejor se 
    # adapten a los datos de calibración. Proporciona también las matrices de
    # proyección. 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, Pmatriz1, dist1, Pmatriz2, dist2, Rmatriz, Tvector, E, F = cv.stereoCalibrate(puntos_calibracion,
                                                                                       puntos_imagenes1,
                                                                                       puntos_imagenes2,
                                                                                       Kmatriz1,
                                                                                       distorsion1,
                                                                                       Kmatriz2,
                                                                                       distorsion2,
                                                                                       (ancho, alto),
                                                                                       criteria = criteria,
                                                                                       flags = stereocalibration_flags)
    return Rmatriz, Tvector
    
def VinculaMatrices(Rmatriz, Tvector, Kmatriz1, Kmatriz2):
    Pmatriz1 = Kmatriz1 @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis = -1)
    Pmatriz2 = Kmatriz2 @ np.concatenate([Rmatriz, Tvector], axis = -1)
    
    return Pmatriz1, Pmatriz2

def Triangulacion(PuntosCamara1, PuntosCamara2, Pmatriz1, Pmatriz2):
    PuntosTriangulados = []
    for Punto1, Punto2 in zip(PuntosCamara1, PuntosCamara2):
        A = [Punto1[1]*Pmatriz1[2,:] - Pmatriz1[1,:],
             Pmatriz1[0,:] - Punto1[0]*Pmatriz1[2,:],
             Punto2[1]*Pmatriz2[2,:] - Pmatriz2[1,:],
             Pmatriz2[0,:] - Punto2[0]*Pmatriz2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        CorrMatriz = A.transpose() @ A
        U, S, VT = linalg.svd(CorrMatriz, full_matrices = False)
        
        aux = VT[3,0:3]/VT[3,3]
        PuntosTriangulados.append(aux)
        
    return np.array(PuntosTriangulados)

def PredecirFrame(cam, modelo):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    imagen = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = modelo.process(imagen)
    imagen.flags.writeable = True
    imagen = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    
    return resultados, imagen

def AbrirVideo(ruta):
    cam = cv.VideoCapture(ruta)
    Nframes = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cam.get(cv.CAP_PROP_FPS)
    
    return cam, Nframes, fps
    
        
        