import sys
sys.path.insert(0, './yolo')

import os
from models import Yolov4
from PIL import Image

RESULT_PATH = 'result/'
PROCESS_PATH = 'process/'


model = Yolov4(class_name_path='data/classes.txt')
model.load_model('yolo/model')


files = os.listdir(PROCESS_PATH)

for f in files:
    output_img, detections = model.predict(PROCESS_PATH + f, random_color=False, show_text=False, plot_img=False)
    #detections = model.predict_raw(PROCESS_PATH + f)
    im = Image.fromarray(output_img)
    im.save(RESULT_PATH + f[:-4] + "_result.png")
    detections.to_csv(RESULT_PATH + f[:-4] + "_boxes.csv", index=False)

