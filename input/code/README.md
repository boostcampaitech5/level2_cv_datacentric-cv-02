## Baseline 코드 사용법 (2023-05-26)
### Settings
1. Repository clone `git clone ~`
2. 기본으로 제공되는 data 폴더를 `input/` 아래로 복사합니다.
3. `dataset.py`, `detect.py`, `deteval.py`, `east_dataset.py`, `inference.py`, `loss.py`, `model.py`를 input/code 아래로 복사합니다.
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
settings: # 실험을 위해 기본적으로 세팅하는 값들입니다.
  who: baseline # who에는 지난 번 대회처럼 이니셜을 넣어주시면, wandb와 모델을 저장할 때 이름이 들어갈 겁니다.
  seed: 42 # 시드입니다.
training: # training에 사용되는 값들입니다. 제가 추가한 건 없고, 원래 있던 것들을 yaml 파일로 옮기기만 했습니다.
  data_dir: ../data/medical
  model_dir: ./trained_models
  num_workers: 8
  image_size: 2048
  input_size: 1024
  batch_size: 8
  learning_rate: 0.001
  max_epoch: 150
  save_interval: 5
  ignore_tags: ['masked', 'excluded-region', 'maintable', 'stamp']
evaluation: # 추후 validation split을 했을 때, evaulation을 위한 hyper-parameters를 적을 생각입니다.
inference: # inference.py를 실행할 때 사용할 값들을 적을 생각입니다.
```
## Training
- 세팅을 완료하고, yaml 파일을 생성하셨다면 training이 가능합니다. Training 방법은 간단합니다.
- 터미널에 `python train.py --config_path ./configs/your_folder/your_yaml_file.yaml` 을 입력하고, 실행하시면 됩니다.
- 학습 과정은 "Optical Character Recognition'이라는 프로젝트 이름으로 WandB에 기록되며,<br>
현재 Epoch, Total Loss, CLS Loss, Angle Loss, IoU Loss를 확인하실 수 있습니다.
- Epoch을 기록하는 이유는, WandB의 그래프 x축은 기본 step 수로 세팅되어 있습니다. 이를 epoch 별로 확인하기 위해 기록합니다.
WandB 상의 그래프 x축을 변경하는 방법은, 그래프 우상단에 있는 edit 아이콘을 누르시고 x축을 epoch으로 변경해주시면 됩니다.
## Inference
- 업데이트 예정입니다.
## Visualization
- Test 이미지 100장에 대해 모델이 예측한 값을 시각화하고, `./viz_results`라는 폴더를 생성한 뒤 하위 파일들로 저장하는 코드입니다.
- 보다 깔끔한 인터페이스를 만들기 위해 업데이트할 예정입니다.
## Others
- 코드를 사용하시다가 추가하고 싶은 기능이 있으시다면 PR을, 버그가 있다면 issue를 활용해주세요! 감사합니다 🙇
