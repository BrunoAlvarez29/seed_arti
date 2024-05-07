import sys
import warnings
import cv2
import os

sys.path.insert(1, 'detectors')
sys.path.insert(2, 'detectors/yolov7')

from detectors.yolo7 import Yolo7


class Detector:
    def __init__(self, detector_name, weights, data, device):

        print("Detector en construcción............!!!")

        self.model = None

        if detector_name is None:
            warnings.warn('detector_name es un objeto NoneType')
            return

        elif detector_name == 'yolo7':
            self.model = Yolo7(weights=weights,
                               data=data,
                               device=device)

        else:
            warnings.warn('Modelo no disponible')
            return

    def predict(self, input):
        if self.model is None:
            warnings.warn('self.Model es un objeto NoneType, por favor seleccione un modelo disponible')
            return

        predictions = self.model.predict(input)
        return predictions


if __name__ == '__main__':

    img_path = '/home/bruno29/catkin_ws/seedling-net/scripts/capture_images/24_03_13/vertical/rgb/2_5.jpg'
    img = cv2.imread(img_path)
    
    # Calcula las coordenadas del centro de la imagen
    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2
    
    # Define los ajustes independientes del ancho y alto
    left_adjustment = int(width * 0.35)  # Ajuste desde el lado izquierdo
    right_adjustment = int(width * 0.3)  # Ajuste desde el lado derecho
    top_adjustment = int(height * 0.32)  # Ajuste desde la parte superior
    bottom_adjustment = int(height * 0.15)  # Ajuste desde la parte inferior
    
    # Define un área de interés alrededor del centro
    roi_x1 = max(0, center_x - left_adjustment)
    roi_y1 = max(0, center_y - top_adjustment)
    roi_x2 = min(width, center_x + right_adjustment)
    roi_y2 = min(height, center_y + bottom_adjustment)
    
    # Extrae el área de interés
    roi_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
    
    detector = Detector('yolo7', weights='../weights/yolov7-vseed.pt', data='../weights/opt.yaml', device='cuda:0')
    predictions = detector.predict(roi_img)
    
    # Encuentra la predicción con el área más grande
    max_area = 0
    largest_pred = None
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_pred = pred
    
    if largest_pred is not None:
        x1, y1, x2, y2 = largest_pred.bbox
        x1 += roi_x1
        y1 += roi_y1
        x2 += roi_x1
        y2 += roi_y1
        
        # Extrae solo el objeto detectado
        detected_object = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Aplica un filtro de binarización al objeto detectado
        gray = cv2.cvtColor(detected_object, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Guarda la imagen binarizada en el mismo directorio con el mismo nombre
        img_name = os.path.basename(img_path)
        binarized_img_path = '/home/bruno29/catkin_ws/seedling-net/scripts/capture_images/24_03_13/vertical/mask' + img_name
        cv2.imwrite(binarized_img_path, binary)
        
        # Muestra ambas imágenes (objeto detectado y versión binarizada)
        cv2.imshow('Objeto detectado', detected_object)
        cv2.imshow('Objeto binarizado', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
