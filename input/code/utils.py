import torch
import numpy as np

import yaml
from datetime import datetime
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


def get_save_folder_name() -> str:
    """파일을 날짜별로 폴더에 저장하고 싶을 때, 폴더명을 만들어주는 함수입니다.

    Returns:
        str: "2023-05-25"와 같이 폴더명으로 사용할 문자열 반환
    """

    today = datetime.now()
    save_folder_name = f"{today.year}-{today.month}-{today.day}"

    return save_folder_name


def load_config(config_file):
    """정의한 YAML 파일을 불러오는 함수입니다.

    Args:
        config_file : 실험에 필요한 설정들을 저장해둔 yaml 파일
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config