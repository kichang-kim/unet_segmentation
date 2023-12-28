import os
import glob
import fiona
import rasterio
import rasterio.features
import rasterio.mask
import numpy as np
import time
from config import DATA_PATH

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def json_to_mask_for_ap(input_json_path, input_tif_path):

    code_mapping = {10: 10, 20: 20, 30: 30, 40: 40, 50: 50, 60: 60}  # 맵핑한 코드와 배열 인덱스를 매핑 for 항공

    total_count_one = 0
    total_count_two = 0
    total_count_three = 0
    total_count_four = 0
    total_count_five = 0
    total_count_six = 0

    area_one = 0
    area_two = 0
    area_three = 0
    area_four = 0
    area_five = 0
    area_six = 0


    file_extension = ".json"

    # 결과물 TIFF 파일 경로
    output_path = DATA_PATH + "/masks"
    # print("DATA_PATH = ", DATA_PATH)

    no_water_json = ""
    invalid_label_json = ""
    invalid_flag = False

    json_files = glob.glob(os.path.join(input_json_path, f"*{file_extension}"))

    # GeoJSON 파일을 열어서 CODE 별로 마스크 생성
    for json_file in json_files:
        # JSON 파일과 대응되는 TIFF 파일 경로 생성
        # print(json_file)
        tif_file = os.path.join(input_tif_path, os.path.basename(json_file).replace(file_extension, ".tif"))
        output_tiff_path = os.path.join(output_path, os.path.basename(tif_file))
        
        with rasterio.open(tif_file) as src:
            out_meta = src.meta
            mask_shape = (src.height, src.width)
            out_meta.update({'dtype': 'uint8', 'count': 1})  # Grayscale 이미지 메타 데이터 업데이트

            # 결과물 TIFF 파일 생성
            with rasterio.open(output_tiff_path, 'w', **out_meta) as dst:
                
                dst_shape = mask_shape

                # 초기 마스크 배열 생성
                mask_array = np.zeros(dst_shape, dtype=np.uint8)
                temp_count = 0
                count_one = 0
                count_two = 0
                count_three = 0
                count_four = 0
                count_five = 0
                count_six = 0
                # GeoJSON 파일을 다시 열어서 마스크 생성 및 저장
                with fiona.open(json_file, "r") as geojson:
                    # print(json_file)
                    for feature in geojson:
                        geometry = feature["geometry"]
                        if "CODE" in feature["properties"]:
                            feature_code = feature["properties"]["CODE"]
                        elif "code" in feature["properties"]:
                            feature_code = feature["properties"]["code"]
                        else:
                            continue
                        
                        # 코드를 배열 인덱스로 변환
                        if feature_code in code_mapping:
                            index = code_mapping[feature_code]
                            if feature_code == 10:
                                count_one += 1
                            elif feature_code == 20:
                                count_two += 1
                            elif feature_code == 30:
                                count_three += 1
                            elif feature_code == 40:
                                count_four += 1
                            elif feature_code == 50:
                                count_five += 1
                            elif feature_code == 60:
                                count_six += 1
                            # print("index: ", index)
                        else:
                            # 처리하지 않는 코드에 대한 기본 인덱스 설정                        
                            # print("no index code")
                            index = 100
                            invalid_flag = True
                            continue

                        # 마스크 생성 (crop=False로 설정)
                        mask = rasterio.features.geometry_mask([geometry], src.shape, src.transform, all_touched=False, invert=True)
                        
                        
                        # 해당 코드에 대한 마스크를 기존 배열에 추가
                        mask_array += mask.astype(np.uint8) * int(index)
                        area_one += (mask_array == 10).sum()
                        area_two += (mask_array == 20).sum()
                        area_three += (mask_array == 30).sum()
                        area_four += (mask_array == 40).sum()
                        area_five += (mask_array == 50).sum()
                        area_six += (mask_array == 60).sum()
                        
                mask = (mask_array == 0)
                mask_array[mask] = 100
                mask = np.logical_not(np.isin(mask_array, [0, 10, 20, 30, 40, 50, 60, 100]))
                mask_array[mask] = 60
                # 결과물 TIFF 파일에 각각의 마스크를 밴드로 추가
                dst.write(mask_array, 1)
                # print("\n", json_file)
                # print(np.unique(mask_array))
                if np.unique(mask_array).size < 1:
                    no_water_json += json_file + "\n"
                    
                # print("Count of 내륙습지 segmentation is: ", count_one)
                # print("Count of 강기슭 segmentation is: ", count_two)
                # print("Count of 호소 segmentation is: ", count_three)
                # print("Count of 암벽바위 segmentation is: ", count_four)
                # print("Count of 하천 segmentation is: ", count_five)
                # print("Count of 기타나지 segmentation is: ", count_six)
                
                total_count_one += count_one
                total_count_two += count_two
                total_count_three += count_three
                total_count_four += count_four
                total_count_five += count_five
                total_count_six += count_six
                
        if invalid_flag == True:
            invalid_label_json += json_file + "\n"
        
    print("\n\nJson file with invalid label: \n", invalid_label_json)
    print("\n\nJson file with no class: \n", no_water_json)

    print("\n\nTotal count of 내륙습지 segmentation is: ", total_count_one)
    print("\n\nTotal count of 강기슭 segmentation is: ", total_count_two)
    print("\n\nTotal count of 호소 segmentation is: ", total_count_three)
    print("\n\nTotal count of 암벽바위 segmentation is: ", total_count_four)
    print("\n\nTotal count of 하천 segmentation is: ", total_count_five)
    print("\n\nTotal count of 기타나지 segmentation is: ", total_count_six)

    print("\n\nTotal pixel area of 내륙습지 segmentation is: ", area_one)
    print("\n\nTotal pixel area of 강기슭 segmentation is: ", area_two)
    print("\n\nTotal pixel area of 호소 segmentation is: ", area_three)
    print("\n\nTotal pixel area of 암벽바위 segmentation is: ", area_four)
    print("\n\nTotal pixel area of 하천 segmentation is: ", area_five)
    print("\n\nTotal pixel area of 기타나지 segmentation is: ", area_six)

def json_to_mask_for_os(input_json_path, input_tif_path):
    code_mapping = {10: 10}  # 맵핑한 코드와 배열 인덱스를 매핑 for 위성

    total_seg_count = 0
    pixel_count = 0

    file_extension = ".json"

    no_water_json = ""
    invalid_label_json = ""

    # 결과물 TIFF 파일 경로
    output_path = DATA_PATH + "/masks"
    # print("DATA_PATH = ", DATA_PATH)

    json_files = glob.glob(os.path.join(input_json_path, f"*{file_extension}"))

    # GeoJSON 파일을 열어서 CODE 별로 마스크 생성
    for json_file in json_files:
        print(json_file)
        # JSON 파일과 대응되는 TIFF 파일 경로 생성
        tif_file = os.path.join(input_tif_path, os.path.basename(json_file).replace(file_extension, ".tif"))
        output_tiff_path = os.path.join(output_path, os.path.basename(tif_file))
        invalid_flag = False
        if(os.path.exists(tif_file)):
            with rasterio.open(tif_file) as src:
                out_meta = src.meta
                mask_shape = (src.height, src.width)
                out_meta.update({'dtype': 'uint8', 'count': 1})  # Grayscale 이미지 메타 데이터 업데이트

                # 결과물 TIFF 파일 생성
                with rasterio.open(output_tiff_path, 'w', **out_meta) as dst:
                    
                    dst_shape = mask_shape

                    # 초기 마스크 배열 생성
                    mask_array = np.zeros(dst_shape, dtype=np.uint8)
                    temp_count = 0
                    count_seg = 0
                    # GeoJSON 파일을 다시 열어서 마스크 생성 및 저장
                    with fiona.open(json_file, "r") as geojson:
                        
                        for feature in geojson:
                            geometry = feature["geometry"]
                            if "CODE" in feature["properties"]:
                                feature_code = feature["properties"]["CODE"]
                            elif "code" in feature["properties"]:
                                feature_code = feature["properties"]["code"]
                            else:
                                continue
                            
                            # 코드를 배열 인덱스로 변환
                            if feature_code in code_mapping: 
                                index = code_mapping[feature_code]
                                count_seg += 1
                                # print("index:", index)
                            elif feature_code == 20:
                                # print("index: 2")
                                index = 100
                            else:
                                # 처리하지 않는 코드에 대한 기본 인덱스 설정                        
                                # print("no index code")
                                invalid_flag = True                            
                                continue

                            # 마스크 생성 (crop=False로 설정)
                            mask = rasterio.features.geometry_mask([geometry], src.shape, src.transform, all_touched=False, invert=True)
                            
                            
                            # print("size[1]: ", mask.size[1])

                            # 해당 코드에 대한 마스크를 기존 배열에 추가
                            # if index == 10:
                            mask_array += mask.astype(np.uint8) * int(index)
                            pixel_count += (mask_array == 10).sum()

                    # 결과물 TIFF 파일에 각각의 마스크를 밴드로 추가
                    mask = (mask_array == 0)
                    mask_array[mask] = 100
                    mask = np.logical_not(np.isin(mask_array, [10, 100]))
                    mask_array[mask] = 10
                    dst.write(mask_array, 1)
                    # print("\n", json_file)
                    # print(np.unique(mask_array))
                    if np.unique(mask_array).size < 2:                    
                        no_water_json += json_file + "\n"
                    # print("Count of water segmentation is: ", count_seg)
                    total_seg_count += count_seg
        
        if invalid_flag == True:
            invalid_label_json += json_file + "\n"

    print("\n\nJson file with invalid label: \n", invalid_label_json)
    print("\n\nJson file with no class: \n", no_water_json)
    print("\n\nTotal count of water segmentation is: ", total_seg_count)
    print("\n\nTotal pixel area of water segmentation is: ", pixel_count)

print("Start Rasterizing!!!!\n")

# GeoJSON 파일 경로와 RGB 이미지 파일 경로
input_json_path = DATA_PATH + "/geojson"
input_tif_path = DATA_PATH + "/imgs_all"
print("DATA_PATH is: ", DATA_PATH)

if "AP" in DATA_PATH:
    json_to_mask_for_ap(input_json_path, input_tif_path)
elif "OS" in DATA_PATH:
    json_to_mask_for_os(input_json_path, input_tif_path)
elif "SS" in DATA_PATH:
    json_to_mask_for_os(input_json_path, input_tif_path)
else:
    print("DATA_PATH is wrong!!")