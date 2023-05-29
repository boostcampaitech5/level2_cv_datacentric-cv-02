import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import json, os
import os.path as osp
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    """visualization에 필요한 arguments를 정의하는 함수입니다.
    """
    parser = ArgumentParser()

    parser.add_argument('--json_name', type=str, default='output.json', help='name of json file which was made by inference.py (default: output.json)')
    parser.add_argument('--save_dir', type=str, default='./viz_results', help='path to save your viz results (default: ./viz_results)')

    args = parser.parse_args()

    return args


def save_viz_results(test_dir: str, json_data: str, save_dir: str) -> None:
    """test data에 대해 모델이 예측한 json 파일을 이용하여, test image에 시각화를 적용한 후 저장하는 함수입니다.

    Args:
        test_dir (str): test data가 저장되어 있는 폴더의 경로입니다.
        json_data (str): 불러온 json 파일의 데이터입니다.
        save_dir (str): 시각화한 결과를 저장할 폴더의 경로입니다.
    """
    # json_data에 존재하는 이미지 정보들만 불러오기
    images = json_data['images']
    file_names = images.keys()

    # 이미지 불러오기
    for file_name in tqdm(file_names, desc='saving viz results...'):
        file_path = osp.join(test_dir, file_name)
        image = cv2.imread(file_path)

        # file_name에 대한 points 가져오기
        for points in images[file_name]['words'].values():
            points = points['points']
        
            # 이미지 상에 points를 그리기
            points = np.array(points, dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=1)

        # 이미지 저장하기
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        save_path = osp.join(save_dir, file_name)
        cv2.imwrite(save_path, image)


if __name__ == '__main__':
    args = parse_args()

    # 1. test data가 있는 폴더의 경로를 가져오기
    test_dir = '../data/medical/img/test'

    # 2. 모델이 예측한 json 파일을 불러오기
    json_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions')
    with open(os.path.join(json_dir, args.json_name), 'r') as f:
        json_data = json.load(f)

    # 3. test 이미지에 예측 결과를 시각화하고, 저장하기
    save_viz_results(test_dir, json_data, args.save_dir)
    