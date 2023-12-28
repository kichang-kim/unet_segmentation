import os
import shutil
from config import DATA_PATH

def copy_files_if_exists(source_folder, destination_folder, check_folder):
    # source_folder: A 폴더
    # destination_folder: C 폴더
    # check_folder: B 폴더

    # source_folder 안의 모든 파일에 대해 반복
    for root, dirs, files in os.walk(source_folder):
        # print(root)
        
        for file in files:
            # print(file)
            source_file_path = os.path.join(root, file)  # A 폴더 내의 파일 경로
            destination_file_path = os.path.join(destination_folder, file)  # C 폴더 내의 파일 경로
            check_file_path = os.path.join(check_folder, file)  # B 폴더 내의 파일 경로

            # B 폴더에 동일한 이름의 파일이 있는지 확인
            if os.path.exists(check_file_path):
                # B 폴더에 파일이 있으면 C 폴더로 복사
                shutil.copy(source_file_path, destination_file_path)
                print(f"File '{file}' copied from {source_folder} to {destination_folder}")


# DATA_PATH = '/workspace/unet/data_NR/AP'
# A 폴더, B 폴더, C 폴더 경로 설정
folder_A = DATA_PATH + '/imgs_all'  # A 폴더 경로
folder_B = DATA_PATH + '/masks'  # B 폴더 경로
folder_C = DATA_PATH + '/imgs'  # C 폴더 경로

# 함수 호출하여 파일 복사 실행
copy_files_if_exists(folder_A, folder_C, folder_B)
