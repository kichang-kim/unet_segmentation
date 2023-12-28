TARGET_AREA = "HR"  # 한강
# TARGET_AREA = "NR"  # 낙동강

TARGET_DATA_TYPE = "AP"             # 항공영상
# TARGET_DATA_TYPE = "SS"          # 위성SAR
# TARGET_DATA_TYPE = "OS"          # 위성이미지


BATCH_SIZE = 8

MAX_EPOCHS = 30
EPOCHS = 30
ACC_CUT_TH = 0.99
CGT_EPOCHS = 0

GPUS = "0,1,2,3"

ignore_label = 255

if TARGET_AREA == "HR":
  DATA_ROOT = "/workspace/unet/data_HR"
elif TARGET_AREA == "NR":
  DATA_ROOT = "/workspace/unet/data_NR"
else:
  print("Target area not found!\n")

if TARGET_DATA_TYPE == "AP":  # 항공
  DATA_PATH = DATA_ROOT + "/AP"
  if TARGET_AREA == "HR":
    WEIGHT_PATH = "20231206_1251_HR_AP_best"
  elif TARGET_AREA == "NR":
    WEIGHT_PATH = "/workspace/unet/checkpoints/NR/AP"
  
  label_mapping = {-1: ignore_label, 0: ignore_label,
                   10: 1, # 내륙습지
                   20: 2, # 강기슭
                   30: 3, # 호소
                   40: 4, # 암벽바위
                   50: 5, # 하천
                   60: 6, # 기타나지
                   100: 0,
                   255: ignore_label
                   }

  visible_mapping = {-1:100,
    1: [184, 131, 237],
    2: [16, 64, 178],
    3: [42, 65, 247],
    4: [200, 229, 155],
    5: [191, 255, 255],
    6: [102, 249, 247]
  }

  NUM_CLASSES = 7
  NUM_CHANNELS = 3
  image_endfix_len = 4
  label_endfix = ""

elif TARGET_DATA_TYPE == "SS":  # SAR위성
  DATA_PATH = DATA_ROOT + "/SS"
  if TARGET_AREA == "HR":
    WEIGHT_PATH = "/workspace/unet/checkpoints/HR/SS"
  elif TARGET_AREA == "NR":
    WEIGHT_PATH = "/workspace/unet/checkpoints/NR/SS"
  
  # 001: 수변
  # 002: 비수변

  label_mapping = {-1: ignore_label, 0: ignore_label,
                   10: 1,
                   100: 0,
                   255: ignore_label}

  visible_mapping = {-1:100,
    1: [184, 131, 237]
  }

  NUM_CLASSES = 2
  NUM_CHANNELS = 3
  image_endfix_len = 4
  label_endfix = ""

elif TARGET_DATA_TYPE == "OS":  # 광학위성
  DATA_PATH = DATA_ROOT + "/OS"
  if TARGET_AREA == "HR":
    WEIGHT_PATH = "/workspace/unet/checkpoints/HR/OS"
  elif TARGET_AREA == "NR":
    WEIGHT_PATH = "/workspace/unet/checkpoints/NR/OS"
  # 10: 수변
  # 20: 비수변

  label_mapping = {-1: ignore_label, 0: ignore_label,
                   10: 1,
                   100: 0,
                   255: ignore_label}

  visible_mapping = {-1:100,
    1: [184, 131, 237]
  }

  NUM_CLASSES = 2
  NUM_CHANNELS = 3
  image_endfix_len = 4
  label_endfix = ""

else:
  print("TARGET_DATA_TYPE is wrong !!!!", TARGET_DATA_TYPE)
  print("TARGET_DATA_TYPE is wrong !!!!", TARGET_DATA_TYPE)
  print("TARGET_DATA_TYPE is wrong !!!!", TARGET_DATA_TYPE)

print("labels:", label_mapping)
