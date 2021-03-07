import os
import sys


TYPE = 1
def join_labels(dataset_type, dataset_name):
    if TYPE == 1:
        join_labels1(dataset_type, dataset_name)
    elif TYPE == 2:
        join_labels2(dataset_type, dataset_name)
        

def join_labels1(dataset_type, dataset_name):
  path = 'data/' + dataset_type + '/' + dataset_name + '/'
  dataset_parts = [part for part in os.listdir(path) if os.path.isdir(path + part)]
   
  with open(path + "/" + "type1_annotations.txt", "w") as labels:
    for dataset_part in dataset_parts:
      part_path = path + "/" + dataset_part
      files = os.listdir(part_path)
      for file in files:
        if not ".txt" in file:
          continue
        
        name = file.split(".")[0]
        print("processing: " + name)
        print(dataset_name + "/" + dataset_part + "/" + name + "_normalmap.png ", file=labels, end="")
          
        with open(part_path + "/" + file, "r") as single_labels:
          space = False
          line = single_labels.readline()
          while line:
            coords = line.split(",")
            if len(coords) == 5:
              x_center, y_center, width, height, cls = [int(c) for c in coords]
              x_min = x_center - width // 2
              y_min = y_center - height // 2
              x_max = x_center + width // 2
              y_max = y_center + height // 2
                
              if space:
                print(" ", file=labels, end="")
              space = True 
              print(f"{x_min},{y_min},{x_max},{y_max},{cls}", file=labels, end="")
            line = single_labels.readline()
            
        print("", file=labels)
        
        
def join_labels2(dataset_type, dataset_name):
  path = 'data/' + dataset_type + '/' + dataset_name + '/'
  dataset_parts = [part for part in os.listdir(path) if os.path.isdir(path + part)]
   
  with open(path + "/" + "type2_annotations.txt", "w") as labels:
    for dataset_part in dataset_parts:
      part_path = path + "/" + dataset_part
      files = os.listdir(part_path)
      for file in files:
        if not ".txt" in file:
          continue
        
        name = file.split(".")[0]
        print("processing: " + name)
          
        with open(part_path + "/" + file, "r") as single_labels:
          line = single_labels.readline()
          while line:
            coords = line.split(",")
            if len(coords) == 5:
              x_center, y_center, width, height, cls = [int(c) for c in coords]
              x_min = x_center - width // 2
              y_min = y_center - height // 2
              x_max = x_center + width // 2
              y_max = y_center + height // 2
                
              print(dataset_name + "/" + dataset_part + "/" + name + "_normalmap.png,", file=labels, end="")
              print("{},{},{},{},wheel".format(x_min, y_min, x_max, y_max), file=labels)
            line = single_labels.readline()     
            
    
def process(dataset_type, dataset_name):    
  raw_path = 'data/raw/' + dataset_type + '/' + dataset_name + '/'
  dataset_parts = os.listdir(raw_path)
  
  for dataset_part in dataset_parts:
    export_path = 'data/' + dataset_type + '/' + dataset_name + '/' + dataset_part + '/'
    if not os.path.exists(export_path):
      os.makedirs(export_path, exist_ok=True)
  
    data_path = raw_path + dataset_part + '/'
    files = os.listdir(data_path)
  
    for file in files:
      if ".cogs" in file:  
        name = file.split('.')[0]
        print("processing: " + name)
        if os.name == 'nt':
          os.system('"utils\WCC.exe"' + 
                    ' --boxes ' + 
                    data_path + file + ' ' + 
                    export_path + ' ' + 
                    data_path)
        else:
          pass



if __name__ == "__main__":
  if len(sys.argv) == 1:
    process('train', 'basic')
    join_labels('train', 'basic')
  elif len(sys.argv) == 2:
    join_labels('train', 'basic')
  elif len(sys.argv) == 3:
    process(sys.argv[1], sys.argv[2])
    join_labels(sys.argv[1], sys.argv[2])

