from utils import DataGenerator, read_annotation_lines
from models import Yolov4
from config import yolo_config
import os

train_lines, val_lines = read_annotation_lines('data/annotations.txt', test_size=0.1)
folder_path = 'data/'
class_name_path = 'data/classes.txt'
data_gen_train = DataGenerator(train_lines, class_name_path, folder_path)
data_gen_val = DataGenerator(val_lines, class_name_path, folder_path)

model = Yolov4(weight_path=None, 
               class_name_path=class_name_path)

model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=32, 
          val_data_gen=data_gen_val,
          callbacks=[])
          
          
PROCESS_PATH = 'data/process/'
model.predict(PROCESS_PATH + os.listdir(PROCESS_PATH)[0], random_color=True)
          
model.save_model('model')