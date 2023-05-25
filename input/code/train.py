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
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./configs/base_config.yaml', help='yaml files to train ocr models (default: ./configs/base_config.yaml)')

    args = parser.parse_args()

    return args


def do_training(num_seed, data_dir,
                model_dir, image_size, input_size,
                num_workers, batch_size, learning_rate,
                max_epoch, save_interval, ignore_tags):

    # seed 세팅
    seed_everything(num_seed)

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

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(settings, training, inference):
    do_training(settings['seed'], training['data_dir'],
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
    settings, training, inference = cfgs['settings'], cfgs['training'], cfgs['inference']

    # wandb 실험 이름 설정
    run_name = f"{settings['who']}_{training['max_epoch']}_{training['input_size']}" # e.g., sy_150_1024 (계속 같은 모델을 사용하니 모델 이름은 제외했어요)
    wandb.run.name = run_name

    # 예외 처리
    if training['input_size'] % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    # 실험 시작!
    main(settings, training, inference)
    