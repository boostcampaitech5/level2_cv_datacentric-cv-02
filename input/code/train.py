import os, json
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

from utils import seed_everything, load_config, get_save_folder_name
from inference import do_inference
from deteval import calc_deteval_metrics


def parse_args():
    """training, evaluation에 필요한 yaml 파일의 경로를 가져오기 위해 사용합니다.

    Returns:
        _type_: 사용자가 입력한 arguments를 반환합니다.
    """
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./configs/base_config.yaml', help='yaml files to train ocr models (default: ./configs/base_config.yaml)')

    args = parser.parse_args()

    return args


def do_training(config_path, settings, train, valid):
    # seed 세팅
    seed_everything(settings['seed'])

    # dataset 생성
    train_dataset = SceneTextDataset(
        settings['data_dir'],
        split='train',
        image_size=train['image_size'],
        crop_size=train['input_size'],
        ignore_tags=settings['ignore_tags']
    )

    # dataloader 생성
    train_dataset = EASTDataset(train_dataset)
    num_train_batches = math.ceil(len(train_dataset) / train['batch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train['batch_size'],
        shuffle=True,
        num_workers=train['num_workers']
    )

    # device, model, optimizer, scheduler 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train['learning_rate'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[train['max_epoch'] // 2], gamma=0.1)

    best_hmean = float('-inf')

    for epoch in range(train['max_epoch']):
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0
        
        # train loop
        model.train()
        with tqdm(total=num_train_batches) as pbar:
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
            epoch_loss / num_train_batches, timedelta(seconds=time.time() - epoch_start)))
        
        # wandb logging - train
        wandb.log({
            "Epoch": epoch + 1,
            "Total Loss": epoch_loss / num_train_batches,
            "CLS Loss": epoch_cls_loss / num_train_batches,
            "Angle Loss": epoch_angle_loss / num_train_batches,
            "IoU Loss": epoch_iou_loss / num_train_batches
        })

        # validation
        if (epoch + 1) % train['save_interval'] == 0:
            print('Calculating DetEval Metrics on Validation Datasets...')
            ufo_result_valid = do_inference(model=model, ckpt_fpath=None,
                                            data_dir=settings['data_dir'],
                                            input_size=valid['input_size'],
                                            batch_size=valid['batch_size'],
                                            split='val')
            ufo_result_valid = ufo_result_valid['images']
            
            # validation을 하기 위한 형태로 predictions, gts를 변환
            gt_bboxes_dict, pred_bboxes_dict = dict(), dict()
            for image_name in ufo_result_valid.keys():
                gt_bboxes_dict[image_name], pred_bboxes_dict[image_name] = [], []

                # gt json load
                with open(osp.join(settings['data_dir'], 'ufo/val.json'), 'r') as f:
                    gt_anno = json.load(f)
                
                # convert format of ground truths to use deteval.py
                gt_anno = gt_anno['images']
                gt_words = gt_anno[image_name]['words'].values()
                for gt_word in gt_words:
                    gt_points = gt_word['points']
                    gt_bboxes_dict[image_name].append(gt_points)
                
                # convert format of predictions to use deteval.py
                pred_words = ufo_result_valid[image_name]['words'].values()
                for pred_word in pred_words:
                    pred_points = pred_word['points']
                    pred_bboxes_dict[image_name].append(pred_points)

            # validation data에 대한 deteval 계산
            results = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, verbose=True)
            precision, recall, hmean = results['total'].values()
            print('Done!')

            print('-'*52)
            print('F1 Score: {:.4f} | Precision: {:.4f} | Recall: {:.4f}'.format(
                hmean, precision, recall))
            print('-'*52)
            
            # wandb logging - validation
            wandb.log({
                "Epoch": epoch + 1,
                "F1 Score": hmean,
                "Precision": precision,
                "Recall": recall,
            })

            # saveF1 score가 갱신되는 경우 모델을 새로 저장
            if hmean > best_hmean:
                print(f'update best model... before: {best_hmean}, after: {hmean}')
                ckpt_fpath = osp.join(settings['model_dir'], get_save_folder_name(config_path)) + '.pth'
                torch.save(model.state_dict(), ckpt_fpath)
                best_hmean = hmean
                print('Done!')


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
    train, valid, _ = cfgs['train'], cfgs['valid'], cfgs['test']

    # wandb 실험 이름 설정
    run_name = f"{settings['who']}_{train['max_epoch']}_{train['input_size']}" # e.g., sy_150_1024 (계속 같은 모델을 사용하니 모델 이름은 제외했어요)
    wandb.run.name = run_name

    # 예외 처리
    if train['input_size'] % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    # 실험 시작!
    do_training(args.config_path, settings, train, valid)
    # main(args.config_path, settings, train, valid)
    