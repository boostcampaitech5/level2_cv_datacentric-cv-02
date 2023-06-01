## Baseline 코드 사용법 (2023-06-01 update)
### Settings
1. Repository clone `git clone ~`
2. 기본으로 제공되는 data 폴더를 `input/` 아래로 복사합니다.
3. `./pths/`, `dataset.py`, `detect.py`, `deteval.py`, `east_dataset.py`, `loss.py`, `model.py`를 `input/code` 아래로 복사합니다.
```
# 3번까지 수행했을 때 디렉토리 구조

level2_cv_datacentric-cv-02
├── .git
├── .github
├── input
│   ├── code
|   │   ├── configs : 실험에 사용할 yaml 파일들을 모아놓은 폴더
|   │   |   ├── base_config.yaml : 제공받은 코드의 초기 세팅들을 모아놓은 yaml 파일
|   │   |   └── sy : 개인 yaml 폴더. 본인 이름의 이니셜로 폴더를 세팅하시면 됩니다.
|   │   |       └── 01_sy_300_1024.yaml : 실험에 사용할 yaml 파일
|   │   |
|   │   ├── pths : 학습에 필요한 pretrained weights입니다. 기본 코드에 있는 걸 그대로 가져오시면 됩니다.
|   │   |
|   │   ├── detect.py         (slack에 공지된 bug 말고는 수정 및 push **불가능**)
|   │   ├── east_dataset.py   (수정 및 push **불가능**)
|   │   ├── loss.py           (수정 및 push **불가능**)
|   │   ├── model.py          (수정 및 push **불가능**)
|   │   ├── dataset.py        (수정 및 push 가능)
|   │   ├── deteval.py        (수정 및 push 가능)
|   │   ├── inference.py      (수정 및 push 가능)
|   │   ├── requirements.txt  (수정 및 push 가능)
|   │   ├── train.py          (수정 및 push 가능)
|   │   ├── utils.py
|   │   └── visualization.py
|   │
│   └── data : 데이터들을 모아두는 폴더 (push **불가능**)
│
├── .gitignore : commit하지 않을 폴더, 파일들을 기록해두는 곳. 상세 설명은 아래에서 참고해주세요!
└── README.md
```
4. `configs` 폴더 아래에 본인 이름의 이니셜로 폴더를 생성하고, 그 아래에 yaml 파일을 만듭니다.<br>
예를 들자면, `configs/sy/01_sy_300_1024.yaml`와 같이 만들어주시면 됩니다.<br>
yaml 파일의 구조는 다음과 같습니다.
```yaml
# (2023-05-30 updated)
settings: # 실험을 위해 기본적으로 세팅하는 값들입니다.
  who: baseline # who에는 지난 번 대회처럼 이니셜을 넣어주시면, wandb와 모델을 저장할 때 이름이 들어갈 겁니다.
  seed: 42 # 시드입니다.
  data_dir: ../data/medical
  model_dir: ./trained_models
  ignore_tags: ['masked', 'excluded-region', 'maintable', 'stamp']
train: # training에 사용되는 값들입니다. 제가 추가한 건 없고, 원래 있던 것들을 yaml 파일로 옮기기만 했습니다.
  num_workers: 8
  image_size: 2048
  input_size: 1024
  batch_size: 8
  learning_rate: 0.001
  max_epoch: 150
  save_interval: 5
valid: # DetEval을 활용한 evaluation 과정에 필요한 값들입니다. Test와 동일한 환경으로 setting 했습니다.
  num_workers: 4
  input_size: 2048
  batch_size: 4
test: # inference.py를 실행할 때 사용할 값들입니다.
  input_size: 2048
  batch_size: 5
```
5. `.gitignore` 파일에 대해 간단히 설명드리겠습니다.
- '수정 및 push 가능'이라고 적힌 코드들 중에서 아직 수정이 되지 않아 push하지 않은 코드들이 있습니다.
- `dataset.py`, `deteval.py`의 경우 아직은 수정한 사항이 없어 push하지 않았습니다.
- 따라서 `.gitignore` 파일에 해당 모듈들이 추가되어있는 상태입니다. `.gitignore` 파일에 추가되면,<br>
`git status`로 모듈들의 버전을 체크할 때 무시됩니다.
- 해당 모듈들을 수정한 뒤 push하고 싶으시다면, `.gitignore`에 있는 모듈의 이름을 지운 뒤 push하시면 됩니다.<br>
(말로 설명해서 복잡해보일 수도 있는데, 해보면 되게 간단합니다.)
- 이후 **settings 3번**의 복사해야 할 파일에서, 해당 모듈을 지워주세요. 이는 불필요한 과정을 없애기 위함입니다.
## Training
- 세팅을 완료하고, yaml 파일을 생성하셨다면 training이 가능합니다. Training 방법은 간단합니다.
- 터미널에 `python train.py --config_path ./configs/your_folder/your_yaml_file.yaml` 을 입력하고, 실행하시면 됩니다.
- 학습 과정은 "Optical Character Recognition'이라는 프로젝트 이름으로 WandB에 기록되며,<br>
현재 Epoch, Total Loss, CLS Loss, Angle Loss, IoU Loss 및 validation set에 대한 f1 score, recall, precision을 확인하실 수 있습니다.
- Epoch을 기록하는 이유는, WandB의 그래프 x축은 기본 step 수로 세팅되어 있습니다. 이를 epoch 별로 확인하기 위해 기록합니다.
WandB 상의 그래프 x축을 변경하는 방법은, 그래프 우상단에 있는 edit 아이콘을 누르시고 x축을 epoch으로 변경해주시면 됩니다.
- - -
- 학습한 모델은 `./trained_models` 아래에 저장됩니다. 저장 주기는 yaml 파일의 `save_interval`에 따라 결정됩니다.<br>
저장되는 형식은 다음과 같습니다.
```
# (2023-05-30 updated)
trained_models
└── 01_sy_300_1024.pth # configs 폴더에 작성한 yaml 파일의 이름으로 weight file이 저장됩니다.
```
- - -
- DetEval metrics를 계산하는 기능이 `train.py`의 training loop에 추가되었습니다.
- [Pull requests #6](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-02/pull/6)을 참조하시면 DetEval metrics를 계산하는 흐름을 간략히 파악하실 수 있습니다.
- 학습이 불충분한 경우 모델이 예측한 bbox의 개수가 너무 많아, deteval metric 계산에 소요되는 시간이 매우 깁니다. 따라서 20 epoch 이후부터 `save_interval` 주기로 평가하도록 코드가 작성되어 있습니다.
## Inference
- 학습이 완료된 모델과, test data를 기반으로 inference가 가능합니다. 사용법은 다음과 같습니다.
```bash
# Example
python inference.py --config_path ./configs/base_config.yaml --weight_path ./trained_models/base_config.pth
```
- 실행 결과는 다음과 같이 저장됩니다.
```
predictions
├── base_config.csv
└── base_config.json
```
## Visualization
- Test 이미지 100장에 대해 모델이 예측한 값을 시각화하고, `./viz_results`라는 폴더를 생성한 뒤 하위 파일들로 저장하는 코드입니다.
- 사용 방법은 다음과 같습니다.
```bash
# Example
python visualization.py --json_name base_config.json
```
- 실행 결과는 다음과 같이 저장됩니다.
```
viz_results
└── base_config
        ├── img1.jpg
        ├── img2.jpg
        ├── ...
        └── img100.jpg
```
## Others
- 코드를 사용하시다가 추가하고 싶은 기능이 있으시다면 PR을, 버그가 있다면 issue를 활용해주세요! 감사합니다 🙇
