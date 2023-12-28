import os
import shutil
import json

# 텍스트 파일에 JSON 파일 이름이 한 줄씩 기록되어 있다고 가정
text_file_path = "../data_NR/OS/test.lst"  # 텍스트 파일의 경로
output_directory = "../data_NR/OS/test"  # 결과 파일을 저장할 디렉토리 경로

# 결과 디렉토리가 없으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 텍스트 파일을 열어서 각 줄을 읽고 처리
with open(text_file_path, "r") as file:
    for line in file:
        json_file_name = line.strip()  # 각 줄의 내용을 파일 이름으로 사용
        tif_file_name = json_file_name.replace("/geojson/", "/imgs/")
        tif_file_name = tif_file_name.replace(".json", ".tif")
        
        input_json_path = os.path.join(json_file_name)  # JSON 파일이 저장된 경로
        json_base_name = os.path.basename(json_file_name)
        tif_base_name = os.path.basename(tif_file_name)
        output_json_path = os.path.join(output_directory, json_base_name)  # 복사될 경로
        output_tif_path = os.path.join(output_directory, tif_base_name)  # 복사될 경로

        # JSON 파일을 복사
        try:
            shutil.move(input_json_path, output_json_path)
            shutil.move(tif_file_name, output_directory)
            print(f"파일 복사 완료: {json_file_name}")
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {json_file_name}")

print("모든 파일 복사 완료")
