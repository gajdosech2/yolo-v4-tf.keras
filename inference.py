import sys
sys.path.insert(0, './yolo')

import os
from models import Yolov4
from PIL import Image

RESULT_PATH = "data/result/"
PROCESS_PATH = 'data/process/'


model = Yolov4(class_name_path='data/classes.txt')
model.load_model('model')
output_img, detections = model.predict(PROCESS_PATH + os.listdir(PROCESS_PATH)[0], random_color=False, show_text=False)

print(type(detections))
print(detections)

im = Image.fromarray(output_img)
im.save(RESULT_PATH + "result.png")

