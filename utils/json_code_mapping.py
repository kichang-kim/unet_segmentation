import os
import json
from config import DATA_PATH

# 특정 폴더 내의 모든 JSON 파일에 대해 작업을 수행하는 함수
def process_json_files_in_folder_for_ap(folder_path):
    # 폴더 내 모든 파일 및 서브폴더 리스트를 얻습니다.
    for root, dir, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                # JSON 파일 경로를 생성합니다.
                json_file_path = os.path.join(root, filename)
                
                # JSON 파일을 열고 데이터를 읽습니다.
                with open(json_file_path, 'r') as file:
                    # print("######### json_file_name_is #######", file)
                    data = json.load(file)

                # "CODE" 키를 찾아서 값 변환
                if "annotation" in data and "features" in data["annotation"]:
                    for feature in data["annotation"]["features"]:
                        if "properties" in feature and "CODE" in feature["properties"]:
                            code_value = feature["properties"]["CODE"]
                            if code_value in ["511", "612", "712", "613", "711", "623"]:
                                feature["properties"]["CODE"] = {
                                    "511": 10,
                                    "612": 20,
                                    "712": 30,
                                    "613": 40,
                                    "711": 50,
                                    "623": 60
                                }[code_value]
                        elif "properties" in feature and "code" in feature["properties"]:
                            code_value = feature["properties"]["code"]
                            if code_value in ["511", "612", "712", "613", "711", "623"]:
                                feature["properties"]["CODE"] = {
                                    "511": 10,
                                    "612": 20,
                                    "712": 30,
                                    "613": 40,
                                    "711": 50,
                                    "623": 60
                                }[code_value]

                # "CODE" 또는 "code" 키를 찾아서 값 변환
                for key in feature.keys():
                    if key.lower() == "code" and feature[key] in ["511", "612", "712", "613", "711", "623"]:
                        feature[key] = {
                            "511": 10,
                            "612": 20,
                            "712": 30,
                            "613": 40,
                            "711": 50,
                            "623": 60
                        }[feature[key]]

                # 결과를 JSON 파일에 다시 저장
                with open(json_file_path, 'w', encoding="UTF-8") as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)
                    
# 특정 폴더 내의 모든 JSON 파일에 대해 작업을 수행하는 함수
def process_json_files_in_folder_for_os(folder_path):
    # 폴더 내 모든 파일 및 서브폴더 리스트를 얻습니다.
    for root, dir, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                # JSON 파일 경로를 생성합니다.
                json_file_path = os.path.join(root, filename)
                
                # JSON 파일을 열고 데이터를 읽습니다.
                with open(json_file_path, 'r') as file:
                    data = json.load(file)

                # "CODE" 키를 찾아서 값 변환
                if "annotation" in data and "features" in data["annotation"]:
                    for feature in data["annotation"]["features"]:
                        if "properties" in feature and "CODE" in feature["properties"]:
                            code_value = feature["properties"]["CODE"]
                            if code_value in ["001", "002"]:
                                feature["properties"]["CODE"] = {
                                    "001": 10,
                                    "002": 20
                                }[code_value]
                        elif "properties" in feature and "code" in feature["properties"]:
                            code_value = feature["properties"]["code"]
                            if code_value in ["001", "002"]:
                                feature["properties"]["code"] = {
                                    "001": 10,
                                    "002": 20
                                }[code_value]

                # "CODE" 또는 "code" 키를 찾아서 값 변환
                for key in feature.keys():
                    if key.lower() == "code" and feature[key] in ["001", "002"]:
                        feature[key] = {
                            "001": 10,
                            "002": 20
                        }[feature[key]]

                # 결과를 JSON 파일에 다시 저장
                with open(json_file_path, 'w', encoding="UTF-8") as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)

# 특정 폴더 내에서 작업을 수행합니다. 폴더 경로를 필요에 따라 수정하세요.

print("Start Json Code Mapping!!!\n")

folder_path = DATA_PATH + '/geojson'
print("Folder Path is: ", folder_path)

if "AP" in DATA_PATH:
    process_json_files_in_folder_for_ap(folder_path)
elif "OS" in DATA_PATH:
    process_json_files_in_folder_for_os(folder_path)
elif "SS" in DATA_PATH:
    process_json_files_in_folder_for_os(folder_path)
else:
    print("DATA_PATH is wrong!!")
