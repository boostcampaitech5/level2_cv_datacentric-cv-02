import torch
import numpy as np

import yaml
import os.path as osp
import random

def seed_everything(seed:int=42):
    """실험 재현을 위해 seed를 설정하는 함수입니다.

    Args:
        seed (int): seed 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_save_folder_name(yaml_path) -> str:
    """파일을 'yaml 파일의 이름/' 방식으로 폴더에 저장할 때, 폴더명을 만들어주는 함수입니다.

    Args:
        yaml_path (_type_): yaml의 경로입니다. e.g., ./configs/sy/01_sy_300_1024.yaml

    Returns:
        str: "01_sy_300_1024/"와 같이 폴더명으로 사용할 문자열 반환
    """

    yaml_folder_name = yaml_path.split('/')[-1].split('.')[0]

    return yaml_folder_name


def load_config(config_file):
    """정의한 YAML 파일을 불러오는 함수입니다.

    Args:
        config_file : 실험에 필요한 설정들을 저장해둔 yaml 파일
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config


# Test Code
if __name__ == "__main__":
    # get_save_folder_name test
    config_path = "./configs/sy/01_sy_300_1024.yaml"
    print(get_save_folder_name(config_path))