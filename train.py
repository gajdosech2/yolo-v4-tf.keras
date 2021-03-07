import sys
sys.path.insert(0, './yolo')

from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import os

train_lines, val_lines = read_annotation_lines('data/train/basic/type1_annotations.txt', test_size=0.1)
data_path = 'data/train'
class_name_path = 'data/classes.txt'
data_gen_train = DataGenerator(train_lines, class_name_path, data_path)
data_gen_val = DataGenerator(val_lines, class_name_path, data_path)

model = Yolov4(weight_path=None, 
               class_name_path=class_name_path)

model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=64, 
          val_data_gen=data_gen_val,
          callbacks=[])
          
          
model.save_model('yolo/model')