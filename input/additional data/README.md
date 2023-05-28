## 외부 데이터 UFO format으로 변환

```
├── json_path
│   ├── bank :  `금융화 특화 문서 OCR 데이터` json 파일 경로 
│   ├── public : `다양한 형태의 한글 문자 OCR` json 파일 경로
├── make_ufo.py : ufo format으로 json 변환
│
```

### make_ufo.py 

```
parser 
├── `--save_path` : 변환한 json 파일 저장 경로 (default : 현재 디렉토리)
├── `--data_mode` : [bank, public, total], total의 경우 두 데이터를 통합해서 변환 (default : total)
├── `--data_path` : 변환시킬 json 파일 위치 (default : json_path)
```

`주의 사항` : 변환시킬 json 파일의 위치를 수정할 경우 위의 예시 디렉토리 처럼 내부에 `bank` or `public` 하위 디렉토리를 만들어 주어야함