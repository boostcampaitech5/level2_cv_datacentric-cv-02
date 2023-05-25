import os
import os.path as osp
import time
import math
from tqdm import tqdm
from datetime import timedelta
from argparse import ArgumentParser
from pprint import pprint

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from utils import seed_everything, load_config


def parse_args():
    """training, evaluation에 필요한 yaml 파일의 경로를 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./configs/base_config.yaml', help='yaml files to train ocr models (default: ./configs/base_config.yaml)')

    args = parser.parse_args()

    return args


def do_training(user_name, seed, data_dir,
                model_dir, image_size, input_size,
                num_workers, batch_size, learning_rate,
                max_epoch, save_interval, ignore_tags):
    """모델을 학습시키는 함수입니다.

    Args:
        user_name (_type_): 학습시킨 모델의 weights를 저장할 때 사용되는 사용자 이름입니다.
        seed (_type_): 재현 가능성을 위한 seed setting에 사용되는 값입니다.
        data_dir (_type_): 데이터가 저장되어 있는 디렉토리 경로입니다.
        model_dir (_type_): 학습시킨 모델의 weights를 저장할 디렉토리 경로입니다.
        image_size (_type_): 원본 이미지를 resize할 때 사용하는 값입니다.
        input_size (_type_): Random Crop을 적용할 때 사용하는 값입니다.
        num_workers (_type_): 데이터를 불러올 때 사용할 CPU 코어 개수입니다.
        batch_size (_type_): training batch size입니다.
        learning_rate (_type_): optimizer의 learning rate입니다.
        max_epoch (_type_): 최대 epoch을 의미합니다.
        save_interval (_type_): 특정 간격으로 model weights를 저장합니다.
        ignore_tags (_type_): 데이터에 존재하는 tag를 보고, 어떤 tag를 무시할 지 결정합니다.
    """
    # seed 세팅
    seed_everything(seed)

    # dataset 생성
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )

    # dataloader 생성
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # device, model, optimizer, scheduler 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # train loop
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        # wandb logging - train
        wandb.log({
            "Epoch": epoch + 1,
            "Total Loss": epoch_loss / num_batches,
            "CLS Loss": epoch_cls_loss / num_batches,
            "Angle Loss": epoch_angle_loss / num_batches,
            "IoU Loss": epoch_iou_loss / num_batches
        })

        # TODO: train/val split과 evaluation 코드 구현하기

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            model_name = f'{user_name}_{epoch}_{input_size}.pth' # save_interval 구간일 때, 현재 에폭을 기준으로 이름을 지정합니다.
            ckpt_fpath = osp.join(model_dir, model_name)
            torch.save(model.state_dict(), ckpt_fpath)


def main(settings, training, evaluation):
    """do_training을 호출하는 함수입니다. 코드의 가독성을 높이기 위해 사용합니다.

    Args:
        settings (_type_): 실험 환경을 setting하기 위한 값들이 담겨있는 dictionary입니다. 사용자(who), seed 값이 담겨있습니다.
        training (_type_): 학습 시 사용할 값들이 담겨있는 dictionary입니다.
        evaluation (_type_): 평가 시 사용할 값들이 담겨있는 dictionary입니다.
                             현재 버전(23.05.25)으로는 train/val split이 안 되어 있기 때문에, 아무 값도 담겨있지 않습니다.
    """
    do_training(settings['who'], settings['seed'], training['data_dir'],
                training['model_dir'], training['image_size'], training['input_size'],
                training['num_workers'], training['batch_size'], training['learning_rate'],
                training['max_epoch'], training['save_interval'], training['ignore_tags'])


if __name__ == '__main__':
    # wandb 세팅
    wandb.init(project="Optical Character Recognition", reinit=True)

    # config_path 불러오기
    args = parse_args()
    print(f"config_path : {args.config_path}")

    # yaml 파일 불러오기
    cfgs = load_config(args.config_path)
    pprint(cfgs)

    # wandb에 config 업로드하기
    wandb.config.update(cfgs)

    # 불러온 yaml 파일을 사용하기 편리하도록 분리하기
    settings = cfgs['settings']
    training, evaluation, _ = cfgs['training'], cfgs['evaluation'], cfgs['inference']

    # wandb 실험 이름 설정
    run_name = f"{settings['who']}_{training['max_epoch']}_{training['input_size']}" # e.g., sy_150_1024 (계속 같은 모델을 사용하니 모델 이름은 제외했어요)
    wandb.run.name = run_name

    # 예외 처리
    if training['input_size'] % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    # 실험 시작!
    main(settings, training, evaluation)
    