import cv2
import argparse
import pyzed.sl as sl
import sys

parser = argparse.ArgumentParser(description='Guardar imágenes de cámaras en un directorio especificado.')
parser.add_argument('--folder', type=str, default='', help='Nombre del directorio donde se guardarán las imágenes.')
args = parser.parse_args()
FECHA = args.folder if args.folder else "capturas"

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  
init_params.coordinate_units = sl.UNIT.MILLIMETER 

# Inincializar la camara zed V
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(repr(err))
    zed.close()
    sys.exit(1)

# Iniciar la camara monocular H
cam_laptop = cv2.VideoCapture(4)
k = 0
while True:
    # Capturar imagen de la camara izquierda ZED
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        image_zed = sl.Mat()
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        img_zed = image_zed.get_data()
        cv2.imshow('ZED Camera', img_zed)

        # Capturar imagen de la cámara monocular
        ret, img_laptop = cam_laptop.read()
        rows, cols, _ = img_laptop.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1) 
        img_laptop = cv2.warpAffine(img_laptop, M, (cols, rows))
        cv2.imshow('Laptop Camera', img_laptop)

        key = cv2.waitKey(10)

        fila = k//6 + 1
        columna = k%6 + 1

        if key == ord('t'):
            confirmation = input("Do you want to store these images? [yes/no]")  
                
            if confirmation == 'y':
                cv2.imwrite(f"{FECHA}/vertical/rgb/{fila}_{columna}.jpg", img_zed)
                cv2.imwrite(f"{FECHA}/horizontal/rgb/{fila}_{columna}.jpg", img_laptop)
                k += 1
                print("Save it!")

        if key == ord('q'):
            break

zed.close()
cam_laptop.release()
cv2.destroyAllWindows()
