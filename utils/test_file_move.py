import os
import shutil
import json
from config import DATA_PATH

print("Start Test File Copy!!!!\n")

# 텍스트 파일에 JSON 파일 이름이 한 줄씩 기록되어 있다고 가정
text_file_path = DATA_PATH + '/test.lst'  # 텍스트 파일의 경로
output_img_directory = DATA_PATH + '/test_s3/imgs'  # 결과 파일을 저장할 디렉토리 경로
output_mask_directory = DATA_PATH + '/test_s3/masks'  # 결과 파일을 저장할 디렉토리 경로
output_json_directory = DATA_PATH + '/test_s3/geojsons'  # 결과 파일을 저장할 디렉토리 경로

# 결과 디렉토리가 없으면 생성
if not os.path.exists(output_img_directory):
    os.makedirs(output_img_directory)
if not os.path.exists(output_mask_directory):
    os.makedirs(output_mask_directory)
if not os.path.exists(output_json_directory):
    os.makedirs(output_json_directory)
    
# 파일 읽기
with open(text_file_path, 'r') as file:
    # 파일의 모든 줄을 읽어서 리스트로 저장
    lines = file.readlines()

# 변환된 데이터를 저장할 리스트
result = []

# 각 줄을 처리하여 결과 리스트에 추가
for line in lines:
    # 각 줄을 공백을 기준으로 분리하여 리스트에 추가
    elements = line.split()
    result.extend(elements)
    
# 결과를 기존 파일에 저장
with open(text_file_path, 'w') as output_file:
    # 결과 리스트의 각 항목을 파일에 쓰기
    for item in result:
        output_file.write(item + '\n')

# 텍스트 파일을 열어서 각 줄을 읽고 처리
with open(text_file_path, "r") as file:
    for line in file:
        file_name = line.strip()  # 각 줄의 내용을 파일 이름으로 사용
        base_name = os.path.basename(file_name)
        if "masks" in file_name:
            output_path = os.path.join(output_mask_directory, base_name)  # 복사될 경로
            print("masks output_path", output_path)
            file_name_json = file_name.replace('masks','geojson')
            file_name_json = file_name_json.replace('.tif', '.json')
            print("masks file name json: ", file_name_json)
            output_path_json = os.path.join(output_json_directory, base_name.split('.')[0])
            output_path_json = output_path_json + ".json"
            print("masks output path json: ", output_path_json)
        elif "imgs" in file_name:
            output_path = os.path.join(output_img_directory, base_name)  # 복사될 경로
            file_name_json = file_name.replace('imgs','geojson')
            file_name_json = file_name_json.replace('.tif', '.json')
            output_path_json = os.path.join(output_json_directory, base_name.split('.')[0])
            output_path_json = output_path_json + ".json"

        # 파일을 복사
        try:
            shutil.copy(file_name, output_path)
            shutil.copy(file_name_json, output_path_json)            
            # print(f"파일 복사 완료: {file_name}")
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_name}")

print("모든 파일 복사 완료")
