import sys
import cv2
import pyzed.sl as sl
import os

# Función para guardar la imagen
def save_depth_image(depth_map, folder):
    filename = os.path.join(folder, "depth_image.png")
    cv2.imwrite(filename, depth_map)
    print(f"Imagen de profundidad guardada en: {filename}")

output_folder = "/home/bruno29/catkin_ws/seedling-net/modules/cap" 

os.makedirs(output_folder, exist_ok=True)

camera = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.4
init_params.depth_maximum_distance = 0.8
init_params.camera_image_flip = sl.FLIP_MODE.AUTO
init_params.depth_stabilization = 1 
runtime_params = sl.RuntimeParameters(confidence_threshold=50, texture_confidence_threshold=200)

err = camera.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(err)
    sys.exit()

depth_map = sl.Mat()

while True:
    if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(depth_map, sl.VIEW.DEPTH)
        numpy_depth_map = depth_map.get_data()
        cv2.imshow('DEPTH', numpy_depth_map)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): 
            break
        elif key & 0xFF == ord('s'):  # Guardar imagen
            save_depth_image(numpy_depth_map, output_folder)

camera.close()
cv2.destroyAllWindows()
