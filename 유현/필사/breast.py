# import zipfile

# # 압축 해제할 zip 파일 경로
# zip_path = "C:/Users/cyh51/Downloads/breast.zip"

# # 압축 해제할 경로
# extract_path = "C:/Users/cyh51/AIS8/00개인"

# # ZipFile 객체 생성
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     # 압축 해제
#     zip_ref.extractall(extract_path)


# 데이터셋 copy 
import shutil
import os
try:
    dataset_path = "C:/Users/cyh51/Downloads" # 데이터가 있는 폴더 경로 
    shutil.copy(os.path.join(dataset_path, 'breast.zip'), 'C:/Users/cyh51/AIS8/00개인/Breast_Ultrasound 필사') # 데이터를 옮길 폴더
except Exception as err:
    print(str(err))




import os

# 변수를 설정해두면, 디렉토리 설정만 변경하면 프로그램 호환성을 높일 수 있음.
ROOT_DIR = 'C:/Users/cyh51/AIS8/00개인/Breast_Ultrasound 필사'
DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'breast')
CLASSIFICATION_DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'Classification')
CLASSIFICATION_TRAIN_ROOT_DIR = os.path.join(CLASSIFICATION_DATA_ROOT_DIR, 'train')

import shutil
if os.path.exists(DATA_ROOT_DIR):
    shutil.rmtree(DATA_ROOT_DIR)

if os.path.exists(CLASSIFICATION_DATA_ROOT_DIR):
    shutil.rmtree(CLASSIFICATION_DATA_ROOT_DIR)

import zipfile
with zipfile.ZipFile(os.path.join(ROOT_DIR, 'breast.zip'),'r') as target_file:
    target_file.extractall(DATA_ROOT_DIR)


import shutil
import os
import glob

total_file_list = glob.glob(os.path.join(DATA_ROOT_DIR, '*'))

# list comprehension을 이용해 benign, malignant, normal을 원소로 가지는 정답 리스트 생성 
label_name_list = [ file_name.split('/')[-1].strip() for file_name in total_file_list if os.path.isdir(file_name) == True ]

# # 해당 경로에 디렉토리가 없을 경우 디렉토리 생성 
# if not os.path.exists(CLASSIFICATION_DATA_ROOT_DIR):
#     os.mkdir(CLASSIFICATION_DATA_ROOT_DIR)

for label_name in label_name_list:

    src_dir_path = os.path.join(DATA_ROOT_DIR, label_name)
    dst_dir_path = os.path.join(CLASSIFICATION_DATA_ROOT_DIR,
                                'trian'+'/'+label_name)





# mask는 분리하거나 삭제 (segmentatiln을 위한 이미지)
