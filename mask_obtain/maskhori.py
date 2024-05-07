import sys
import warnings
import os
import cv2
import numpy as np

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'detectors')
sys.path.insert(2, 'detectors/yolov7')

from detectors.yolo7 import Yolo7

class Detector:
    def __init__(self, detector_name, weights, data, device):
        
        print("Detector en construcción............!!!")
        
        self.model = None

        if detector_name is None:
            warnings.warn('detector_name es un objeto de tipo NoneType')
            return
        elif detector_name == 'yolo7':
            self.model = Yolo7(weights=weights, data=data, device=device)
        else:
            warnings.warn('El modelo no está disponible')
            return

    def predict(self, img):
        if self.model is None:
            warnings.warn('self.Model es un objeto de tipo NoneType, por favor seleccione un modelo disponible')
            return
        predictions = self.model.predict(img)
        return predictions

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

if __name__ == '__main__':
    img_path = '/home/bruno29/catkin_ws/seedling-net/scripts/capture_images/24_03_13/vertical/rgb/1_3.jpg'
    detector = Detector('yolo7', weights='../weights/yolov7-vseed.pt', data='../weights/opt.yaml', device='cuda:0')
    img = cv2.imread(img_path)

    detection_folder = '/home/bruno29/catkin_ws/seedling-net/scripts/capture_images/24_03_13/vertical/detection'
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)

    specific_height = 180 
    margin = 200  # Aumentado el margen de la región de interés en los lados

    # Definir la región de interés (ROI) con menos altura y más ancho
    height, width, _ = img.shape
    roi_y1 = height // 3
    roi_y2 = 2 * height // 3
    roi_x1 = width // 4 - margin
    roi_x2 = 3 * width // 4 + margin
    roi_img = img[roi_y1:roi_y2, roi_x1:roi_x2]

    cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    predictions = detector.predict(roi_img)

    largest_prediction = None
    max_area = 0

    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox

        x1 += roi_x1
        y1 += roi_y1
        x2 += roi_x1
        y2 += roi_y1
        height = y2 - y1
        if height > specific_height:
            # Calcular el área de la predicción
            area = bbox_area((x1, y1, x2, y2))
            # Actualizar la detección más grande
            if area > max_area:
                max_area = area
                largest_prediction = pred

    # Dibujar solo la detección más grande
    if largest_prediction is not None:
        x1, y1, x2, y2 = largest_prediction.bbox
        x1 += roi_x1
        y1 += roi_y1
        x2 += roi_x1
        y2 += roi_y1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Guardar solo la detección más grande
        file_name = os.path.basename(img_path)
        save_path = os.path.join(detection_folder, file_name)
        cv2.imwrite(save_path, img)

        # Extraer la máscara del plantín
        mask = largest_prediction.mask
        mask_gray = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_detection_folder = '/home/bruno29/catkin_ws/seedling-net/scripts/capture_images/24_03_13/vertical/mask'
        if not os.path.exists(mask_detection_folder):
            os.makedirs(mask_detection_folder)
        mask_filename = f"mask_{file_name}"
        mask_path = os.path.join(mask_detection_folder, mask_filename)
        cv2.imwrite(mask_path, mask_gray)

    cv2.imshow('Deteccion', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()