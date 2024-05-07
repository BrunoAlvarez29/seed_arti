import cv2
import numpy as np

def undistort_image(image_file, camera_matrix, distortion_coefficients):
    # Cargar la imagen
    image = cv2.imread(image_file)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Obtener la altura y el ancho de la imagen
    h, w = image.shape[:2]

    # Generar el mapa de corrección
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, (w, h), 5)
    
    # Aplicar el mapa de corrección a la imagen
    undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # Recortar la imagen para eliminar las áreas negras
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # Mostrar la imagen corregida
    cv2.imshow("Imagen original", image)
    cv2.imshow("Imagen corregida", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Parámetros intrínsecos y extrínsecos de la cámara obtenidos durante la calibración
camera_matrix = np.array([[977.91106995, 0, 342.58569442],
                          [0, 975.46476456, 211.95226346],
                          [0, 0, 1]])
distortion_coefficients = np.array([-0.404861084, -1.39165795, 0.00203628652, -0.0016171769, 20.0575806])

# Ruta de la imagen
image_file = "./24_03_13/horizontal/rgb/1_2.jpg"

# Corregir la distorsión de la imagen y mostrarla
undistort_image(image_file, camera_matrix, distortion_coefficients)
