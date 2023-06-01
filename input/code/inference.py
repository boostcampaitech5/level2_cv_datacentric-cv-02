import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm
from pprint import pprint

from detect import detect

from utils import get_save_folder_name, load_config


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    """inference에 필요한 yaml 파일의 경로를 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--config_path', type=str, default='./configs/base_config.yaml', help='yaml files to train ocr models (default: ./configs/base_config.yaml)')
    parser.add_argument('--weight_path', type=str, required=True, help='trained model weights to inference test data')

    args = parser.parse_args()

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    if split == 'test': # test인 경우만 weight를 불러옵니다.
        model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, 'img/{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args, settings, inference):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = EAST(pretrained=False).to(device)

    # Get paths to checkpoint files
    ckpt_fpath = args.weight_path

    if not osp.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, settings['data_dir'], inference['input_size'],
                                inference['batch_size'], split='test')
    ufo_result['images'].update(split_result['images'])

    # TODO: json 파일로 저장할 수 있게 코드 추가하기
    output_json_name = ckpt_fpath.split('.')[1].split('/')[-1] + '.json'
    with open(osp.join(settings['output_dir'], output_json_name), 'w') as f:
        json.dump(ufo_result, f, indent=4)

    output_fname = ckpt_fpath.split('.')[1].split('/')[-1] + '.csv'
    with open(osp.join(settings['output_dir'], output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    # 사용자 입력 argument
    args = parse_args()
    pprint(args)

    # yaml 파일 불러오기
    cfgs = load_config(args.config_path)
    pprint(cfgs)

    # 불러온 yaml 파일을 사용하기 편리하도록 분리하기
    settings = cfgs['settings']
    training, evaluation, inference = cfgs['train'], cfgs['valid'], cfgs['test']

    # 예외 처리
    if inference['input_size'] % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    # inference 시작!
    main(args, settings, inference)
