import os
import shutil
from config import DATA_PATH

print("Making Train/Valid/Test Dataset List\n")

images_dataset_folder = DATA_PATH + '/imgs'
# new_images_dataset_folder = DATA_PATH + '/imgs_new'
label_dataset_folder = DATA_PATH + '/masks' 
root_foler = DATA_PATH

test_percent = 0.1
val_percent = 0.1

n_val = int(len(os.listdir(label_dataset_folder)) * val_percent)
n_test = int(len(os.listdir(label_dataset_folder)) * test_percent)
n_train = int(len(os.listdir(label_dataset_folder)) - n_val - n_test)
print("n_val: ", n_val)
print("n_test: ", n_test)
print("n_train: ", n_train)

with open(root_foler + '/train.lst', 'w') as train_file:
    with open(root_foler + '/val.lst', 'w') as val_file:
        with open(root_foler + '/test.lst', 'w') as test_file:
            count = 0
            for filename in os.listdir(label_dataset_folder):
                name, extension = os.path.splitext(filename)
                
                if extension == '.tif':
                    tif_filename = name + '.tif'
                    tif_path = os.path.join(images_dataset_folder, tif_filename)
                    # new_tif_path = os.path.join(new_images_dataset_folder, tif_filename)
                    mask_path = os.path.join(label_dataset_folder, filename)
                    
                    if os.path.exists(tif_path):
                        # shutil.move(os.path.abspath(tif_path), os.path.abspath(new_tif_path)) ### 데이터셋 변경 있을 때 주석 해제
                        if count < n_train:
                            train_file.write(os.path.abspath(tif_path) + " " + os.path.abspath(mask_path)+"\n")
                        elif (count >= n_train) & (count < (n_train + n_val)):
                            val_file.write(os.path.abspath(tif_path) + " " + os.path.abspath(mask_path)+"\n")
                        else:
                            test_file.write(os.path.abspath(tif_path) + " " + os.path.abspath(mask_path)+"\n")
                count += 1