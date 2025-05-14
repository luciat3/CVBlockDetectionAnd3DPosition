# -*- coding: utf-8 -*-
"""
Autors: Lucía Torrescusa Rubio i Joel Montes de Oca
"""
# -------------------------
# IMPORTS
# -------------------------

import numpy as np
#import sympy as sp
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
import sim
import time
import random
import math



# -------------------------
# VARIABLES GLOBALES
# -------------------------

# - Dimensiones plano
PLANE_WIDTH = 0.500
PLANE_HEIGHT = 0.500
PLANE_MARGIN = 0.005  # 5mm de margen
PLANE_CENTER = [0.0, 0.0, 0.002]  # Coordenadas plano

# - Tamaño de bloque de Jenga
BLOQUE_SIZE_X = 0.075
BLOQUE_SIZE_Y = 0.025
BLOQUE_SIZE_Z = 0.015

separacion = 0.001
POS_Z = 0.02


def conectar():
    sim.simxFinish(-1)  # cerrar posibles conexiones previas
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print("Conectado a CoppeliaSim")
    else:
        raise Exception("No se pudo conectar a CoppeliaSim")
    return clientID

def obtener_handles(clientID, tolerancia=0.005):
    cubos = []
    err, all_shapes = sim.simxGetObjects(clientID, sim.sim_object_shape_type, sim.simx_opmode_blocking)
    if err != 0:
        raise Exception(" No se pudieron obtener las piezas")

    for obj in all_shapes:
        # Parámetros de tamaño del bounding box (mitades)
        res_x, max_x = sim.simxGetObjectFloatParameter(clientID, obj, 18, sim.simx_opmode_blocking)
        res_y, max_y = sim.simxGetObjectFloatParameter(clientID, obj, 19, sim.simx_opmode_blocking)
        res_z, max_z = sim.simxGetObjectFloatParameter(clientID, obj, 20, sim.simx_opmode_blocking)

        if res_x == 0 and res_y == 0 and res_z == 0:
            size_x = max_x * 2
            size_y = max_y * 2
            size_z = max_z * 2

            # Comparación con tamaños del bloque de Jenga
            if (
                abs(size_x - 0.075) < tolerancia and
                abs(size_y - 0.025) < tolerancia and
                abs(size_z - 0.015) < tolerancia
            ):
                cubos.append(obj)

    print(f" Detectados {len(cubos)} bloques de Jenga")
    return cubos

def capturar_imagen(clientID, nombre_camara, nombre_archivo, carpeta):
    
    # Crear carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)    
    
    # Obtener handle de la cámara
    retCode, sensorHandle = sim.simxGetObjectHandle(clientID, nombre_camara, sim.simx_opmode_blocking)
    if retCode != 0:
        raise Exception(f" No se pudo obtener el handle de la cámara {nombre_camara}")
    
    # Capturar imagen
    retCode, resolution, image = sim.simxGetVisionSensorImage(clientID, sensorHandle, 0, sim.simx_opmode_oneshot_wait)
    if retCode != 0:
        raise Exception(f" No se pudo capturar imagen de la cámara {nombre_camara}")
    
    # Procesar imagen
    img = np.array(image, dtype=np.uint8)
    img.resize([resolution[1], resolution[0], 3])
    img = np.flip(img, axis=0)  # Para corregir que OpenCV y CoppeliaSim tienen coordenadas diferentes

    # Mostrar imagen
    plt.imshow(img)
    plt.title(nombre_camara + "-" + nombre_archivo)
    plt.axis('off')
    plt.show()

    # Guardar imagen
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    cv2.imwrite(ruta_completa, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



# -------------------------
# FUNCIONES PARA LAS POSICIONES DE PIEZAS 
# -------------------------

def obtener_vertices(x, y, w, h, angulo):
    """
    Devuelve los 4 vértices de un rectángulo rotado.
    """
    cos_a = math.cos(angulo)
    sin_a = math.sin(angulo)

    dx = w / 2
    dy = h / 2

    # Definir los 4 vértices relativos al centro (antes de rotar)
    vertices = [
        ( dx,  dy),
        (-dx,  dy),
        (-dx, -dy),
        ( dx, -dy)
    ]

    # Rotar y trasladar
    vertices_rotados = []
    for (vx, vy) in vertices:
        x_rot = x + (vx * cos_a - vy * sin_a)
        y_rot = y + (vx * sin_a + vy * cos_a)
        vertices_rotados.append((x_rot, y_rot))

    return vertices_rotados

def proyectar(vertices, eje):
    """
    Proyecta un conjunto de vértices sobre un eje.
    Devuelve el mínimo y máximo escalar proyectado.
    """
    min_proj = max_proj = None
    for (x, y) in vertices:
        proyeccion = x * eje[0] + y * eje[1]
        if (min_proj is None) or (proyeccion < min_proj):
            min_proj = proyeccion
        if (max_proj is None) or (proyeccion > max_proj):
            max_proj = proyeccion
    return min_proj, max_proj


def colisionan(pieza1, pieza2):
    """
    Comprueba si dos piezas (x, y, angle) colisionan usando SAT.
    """
    x1, y1, ang1 = pieza1
    x2, y2, ang2 = pieza2

    w = BLOQUE_SIZE_X + separacion
    h = BLOQUE_SIZE_Y + separacion

    # Obtener los vértices de los dos rectángulos
    vertices1 = obtener_vertices(x1, y1, w, h, ang1)
    vertices2 = obtener_vertices(x2, y2, w, h, ang2)

    # Ejes a comprobar = normales a los lados de los rectángulos
    ejes = []
    for i in range(4):
        # Cada lado es el vector entre dos vértices
        p1 = vertices1[i]
        p2 = vertices1[(i+1)%4]
        borde = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal (perpendicular)
        normal = (-borde[1], borde[0])
        # Normalizar
        mag = math.hypot(*normal)
        normal = (normal[0]/mag, normal[1]/mag)
        ejes.append(normal)

    for i in range(4):
        p1 = vertices2[i]
        p2 = vertices2[(i+1)%4]
        borde = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-borde[1], borde[0])
        mag = math.hypot(*normal)
        normal = (normal[0]/mag, normal[1]/mag)
        ejes.append(normal)

    # Para cada eje, proyectar los dos rectángulos y comprobar solapamiento
    for eje in ejes:
        min1, max1 = proyectar(vertices1, eje)
        min2, max2 = proyectar(vertices2, eje)

        if max1 < min2 or max2 < min1:
            # Hay una separación en este eje → no colisionan
            return False

    # No hubo separación en ningún eje → colisionan
    return True


# -------------------------
# FUNCIONES NIVELES
# -------------------------

def nivel_ordenado(clientID, cubos, num_imagenes=10):
    
    print(f"Nivel simple: generando {num_imagenes} imágenes")
     
    for img_idx in range(num_imagenes):
        
        print(f"Generando imagen {img_idx+1}/{num_imagenes}")
        
        posiciones = []
        intentos = 0
        max_intentos = 10000
        
        while len(posiciones) < len(cubos) and intentos < max_intentos:
            intentos += 1
            
            x = random.uniform(-PLANE_WIDTH/2 + BLOQUE_SIZE_X/2, PLANE_WIDTH/2 - BLOQUE_SIZE_X/2)
            y = random.uniform(-PLANE_HEIGHT/2 + BLOQUE_SIZE_Y/2, PLANE_HEIGHT/2 - BLOQUE_SIZE_Y/2)
            nueva_pieza = (x, y)
            
            # Comprobar si la nueva pieza colisiona con las existentes
            colision = False
            for (px, py) in posiciones:
                if (x < px + BLOQUE_SIZE_X + separacion and
                    x + BLOQUE_SIZE_X + separacion > px and
                    y < py + BLOQUE_SIZE_Y + separacion and
                    y + BLOQUE_SIZE_Y + separacion > py):
                    colision = True
                    break

            if not colision:
                posiciones.append(nueva_pieza)
                
        if intentos >= max_intentos:
            print(f"Advertencia: No se pudo generar imagen {img_idx+1} en {max_intentos} intentos")
            continue
        
                
        for idx, cubo in enumerate(cubos):
            pos_x, pos_y = posiciones[idx]
            
            sim.simxSetObjectPosition(clientID, cubo, -1, [pos_x, pos_y, POS_Z], sim.simx_opmode_oneshot)
            sim.simxSetObjectOrientation(clientID, cubo, -1, [0, 0, 0], sim.simx_opmode_oneshot)
        
        time.sleep(0.5)

        # Guardar imágenes
        nombre_base = f"nSimple_{img_idx:03d}"
        capturar_imagen(clientID, "zenital", nombre_base + "_zenital.png", "./dataset/nivelSimple/zenital/")
        capturar_imagen(clientID, "lateral", nombre_base + "_lateral.png", "./dataset/nivelSimple/lateral/")




#Este nivel es más complejo de comprobar colisiones
#Usaremos el algoritmo Separating Axis Theorem
#Y las funciones auxiliares obtener_vertices, proyectar y colisionan

def nivel_orientaciones(clientID, cubos, num_imagenes=10):
    print(f" Nivel orientaciones aleatorias: generando {num_imagenes} imágenes")
    for img_idx in range(num_imagenes):
            print(f"Generando imagen {img_idx+1}/{num_imagenes}")
            
            posiciones = []
            intentos = 0
            max_intentos = 10000
            
            while len(posiciones) < len(cubos) and intentos < max_intentos:
                intentos += 1
                
                x = random.uniform(-PLANE_WIDTH/2 + BLOQUE_SIZE_X/2, PLANE_WIDTH/2 - BLOQUE_SIZE_X/2)
                y = random.uniform(-PLANE_HEIGHT/2 + BLOQUE_SIZE_Y/2, PLANE_HEIGHT/2 - BLOQUE_SIZE_Y/2)
                angulo_z = random.uniform(0, 2 * math.pi)
    
                nueva_pieza = (x, y, angulo_z)
    
                # Comprobar colisiones con las piezas existentes
                colision = False
                for pieza in posiciones:
                    if colisionan(nueva_pieza, pieza):
                        colision = True
                        break
    
                if not colision:
                    posiciones.append(nueva_pieza)
    
            if intentos >= max_intentos:
                print(f"Advertencia: No se pudo generar imagen {img_idx+1} en {max_intentos} intentos")
                continue
    
            # Colocar los cubos
            for idx, cubo in enumerate(cubos):
                pos_x, pos_y, angulo_z = posiciones[idx]
                sim.simxSetObjectPosition(clientID, cubo, -1, [pos_x, pos_y, POS_Z], sim.simx_opmode_oneshot)
                sim.simxSetObjectOrientation(clientID, cubo, -1, [0, 0, angulo_z], sim.simx_opmode_oneshot)
    
            time.sleep(0.5)
    
            # Guardar imágenes
            nombre_base = f"nOrientaciones_{img_idx:03d}"
            capturar_imagen(clientID, "zenital", nombre_base + "_zenital.png", "./dataset/nivelOrientaciones/zenital/")
            capturar_imagen(clientID, "lateral", nombre_base + "_lateral.png", "./dataset/nivelOrientaciones/lateral/")

def nivel_solapado(clientID, cubos, num_imagenes=10):
    print(f" Nivel solapado: generando {num_imagenes} imágenes")
    
    ALTURA_BLOQUE = 0.05  # Altura de cada bloque para apilar visualmente
    MAX_NIVEL_APILADO = 3  # Máximo número de bloques uno encima de otro

    for img_idx in range(num_imagenes):
        print(f"Generando imagen {img_idx+1}/{num_imagenes}")
        
        posiciones = []
        
        for cubo in cubos:
            x = random.uniform(-PLANE_WIDTH/2 + BLOQUE_SIZE_X/2, PLANE_WIDTH/2 - BLOQUE_SIZE_X/2)
            y = random.uniform(-PLANE_HEIGHT/2 + BLOQUE_SIZE_Y/2, PLANE_HEIGHT/2 - BLOQUE_SIZE_Y/2)
            angulo_z = random.uniform(0, 2 * math.pi)

            nivel_apilado = random.randint(0, MAX_NIVEL_APILADO - 1)  # Cuánto subimos el bloque en Z
            z = POS_Z + nivel_apilado * ALTURA_BLOQUE

            posiciones.append((x, y, z, angulo_z))

        # Colocar los cubos
        for idx, cubo in enumerate(cubos):
            pos_x, pos_y, pos_z, angulo_z = posiciones[idx]
            sim.simxSetObjectPosition(clientID, cubo, -1, [pos_x, pos_y, pos_z], sim.simx_opmode_oneshot)
            sim.simxSetObjectOrientation(clientID, cubo, -1, [0, 0, angulo_z], sim.simx_opmode_oneshot)

        time.sleep(0.5)

        # Guardar imágenes
        nombre_base = f"nSolapado_{img_idx:03d}"
        capturar_imagen(clientID, "zenital", nombre_base + "_zenital.png", "./dataset/nivelSolapado/zenital/")
        capturar_imagen(clientID, "lateral", nombre_base + "_lateral.png", "./dataset/nivelSolapado/lateral/")

def nivel_caotico(clientID, cubos, num_imagenes=10):
    print(" Nivel caótico: generando {num_imagenes} imágenes")
    

    ALTURA_BLOQUE = 0.05  # Altura de cada bloque
    RADIO_MONTON = 0.05   # Radio pequeño alrededor del centro
    POS_Z_BASE = 0.02     # Base de altura en Z

    for img_idx in range(num_imagenes):
        print(f"Generando imagen {img_idx+1}/{num_imagenes}")
        
        posiciones = []

        for cubo in cubos:
            # X e Y cercanos a (0,0)
            x = random.uniform(-RADIO_MONTON, RADIO_MONTON)
            y = random.uniform(-RADIO_MONTON, RADIO_MONTON)
            angulo_z = random.uniform(0, 2 * math.pi)

            # Subimos aleatoriamente cada cubo
            nivel_apilado = random.randint(0, 5)  # Puedes ajustar este rango para más o menos altura
            z = POS_Z_BASE + nivel_apilado * ALTURA_BLOQUE

            posiciones.append((x, y, z, angulo_z))

        # Colocar los cubos
        for idx, cubo in enumerate(cubos):
            pos_x, pos_y, pos_z, angulo_z = posiciones[idx]
            sim.simxSetObjectPosition(clientID, cubo, -1, [pos_x, pos_y, pos_z], sim.simx_opmode_oneshot)
            sim.simxSetObjectOrientation(clientID, cubo, -1, [0, 0, angulo_z], sim.simx_opmode_oneshot)

        time.sleep(0.5)

        # Guardar imágenes
        nombre_base = f"nCaotico_{img_idx:03d}"
        capturar_imagen(clientID, "zenital", nombre_base + "_zenital.png", "./dataset/nivelCaotico/zenital/")
        capturar_imagen(clientID, "lateral", nombre_base + "_lateral.png", "./dataset/nivelCaotico/lateral/")





# -------------------------
# EJECUCIÓN
# -------------------------
if __name__ == '__main__':
    clientID = conectar()
    time.sleep(1)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    time.sleep(1)
    
    
    cubos = obtener_handles(clientID)

    # ---------------- ORDENADO ----------------
    nivel_ordenado(clientID, cubos, 10)

    # ---------------- ORIENTACIONES ----------------
    nivel_orientaciones(clientID, cubos, 10)

    # ---------------- SOLAPADO ----------------
    nivel_solapado(clientID, cubos, 10)
    
    # ---------------- CAOTICO ----------------
    nivel_caotico(clientID, cubos, 10)
    

    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
    sim.simxFinish(clientID)
