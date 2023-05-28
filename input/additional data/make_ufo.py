import os
from glob import glob
import pprint

import json
from argparse import ArgumentParser
from collections import defaultdict



def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--save_path', type=str, default='./', help='새롭게 만든 json 파일 저장 위치')
    
    parser.add_argument('--data_mode', type=str, default='total', help='UFO 형식으로 변환시킬 데이터의 종류 [public, bank, total:둘다]')
    parser.add_argument('--data_path', type=str, default='./json_path/', help='변환시킬 json 파일 위치')
    args = parser.parse_args()

    return args

def public_json(json_list, UFO):
    """_summary_

    Args:
        json_list (list_): json file path list
        UFO (defaultdict):  default dict 

    Returns:
        UFO (defaultdict): Updated Dict file
    """
    for json_file in json_list:
        with open(json_file, encoding='utf-8') as f:
            json_data = json.load(f)

        temp = defaultdict(dict)   
        
        temp[json_data['image']['file_name']] = {
            'paragraphs' : {},
            'words' : {
                
            },
            'chars' : {},
            'img_w' : json_data['image']['width'], 
            'img_h' : json_data['image']['height'],
        }
        
        for idx, word in enumerate(list(json_data['text']['word'])):
            temp[json_data['image']['file_name']]['words'][str(idx+1).zfill(4)] = {
                'transcription' : word['value'],
                'points' : [[word['wordbox'][0], word['wordbox'][3]],
                            [word['wordbox'][2], word['wordbox'][3]],
                            [word['wordbox'][2], word['wordbox'][1]],
                            [word['wordbox'][0], word['wordbox'][1]]],
                'orientation' : "Horizontal",
                'language': None,
                "tags": ["Auto"],
                "confidence": None,
                "illegibility": False
            }

        UFO['images'].update(temp)

    return UFO

def bank_json(json_list, UFO):
    """_summary_

    Args:
        json_list (list_): json file path list
        UFO (defaultdict):  default dict 

    Returns:
        UFO (defaultdict): Updated Dict file
    """
    for json_file in json_list:
        with open(json_file, encoding='utf-8') as f:
            json_data = json.load(f)

        temp = defaultdict(dict)   

        temp[json_data['name']] = {
            'paragraphs' : {},
            'words' : {
                
            },
            'chars' : {},
            'img_w' : json_data['images'][0]['width'], 
            'img_h' : json_data['images'][0]['height'],
        }
        idx = 1
        for word in list(json_data['annotations']):
            for poly in word['polygons']:
                temp[json_data['name']]['words'][str(idx).zfill(4)] = {
                    'transcription' : poly['text'],
                    'points' : [
                        poly['points'][0],
                        poly['points'][3],
                        poly['points'][2],
                        poly['points'][1]
                        ],
                    'orientation' : "Horizontal",
                    'language': None,
                    "tags": ["Auto"],
                    "confidence": None,
                    "illegibility": False
                }
                idx+=1
        UFO['images'].update(temp)
        
        
    return UFO


def main(args):

    bank_path = os.path.join(args.data_path, 'bank')
    public_path = os.path.join(args.data_path, 'public')

    bank_list = glob(os.path.join(bank_path, '*.json'))
    public_list = glob(os.path.join(public_path, '*.json'))

    temp_UFO1 = defaultdict(dict)
    temp_UFO2 = defaultdict(dict)
    public_ufo = public_json(public_list,temp_UFO1)
    bank_ufo = bank_json(bank_list, temp_UFO2)
    total_ufo = bank_json(bank_list, public_ufo)

    if args.data_mode == 'total':
        with open(os.path.join(args.save_path, f'{args.data_mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(total_ufo, f, ensure_ascii=False, indent=4)
    elif args.data_mode == 'public':
        
        with open(os.path.join(args.save_path, f'{args.data_mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(public_ufo, f, ensure_ascii=False, indent=4)
    elif args.data_mode == 'bank':
        
        with open(os.path.join(args.save_path, f'{args.data_mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(bank_ufo, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    args = parse_args()
    main(args)
