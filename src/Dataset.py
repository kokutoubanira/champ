import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image

import os.path as osp
import glob


#入力画像の処理を行う
#訓練時と推論時で処理が異なる

class ImageTransform():
    """
    画像の前処理クラス。
    画像のサイズをリサイズ
    resize: int
        リサイズ先の画像の大きさ
    mean : (RGB)
    std:(RGB)
    """

    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize), #リサイズ
                # transforms.RandomRotation(degrees=20), #ランダムに回転
                transforms.ToTensor(), #テンソルに変換
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize), #リサイズ
                transforms.ToTensor(), #テンソルに変換
            ])
        }

    def __call__(self, img, phase='train'):
        """
        pahes:'train' or 'val'
        """
        return self.data_transform[phase](img)


class MyDataset(data.Dataset):
    '''
    file_list : リスト
        画像パス
    transform: object
        前処理クラスのインスタンス
    phase : 学習化テストか設定する
    '''
    def __init__(self, file_list, transform=None, phase='train'):
        self.transform = transform#前処理クラスのリスト
        self.file_list = file_list#ファイルパスのリスト
        self.phase = phase#train or val の指定

    def __len__(self):
        #画像の枚数を返す
        return len(self.file_list)
    
    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルの取得
        '''
        #index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        t_data = Image.new("RGB", img.size, (255, 255, 255))
        t_data.paste(img, mask=img.split()[3])
        if t_data.mode == "RGB":
            t_data = t_data.convert("L")

        #画像の前処理実施
        img_transformed = self.transform(
            t_data, self.phase
        )

        #画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            label = int(img_path[19])
        elif self.phase == "val":
            label = int(img_path[15])
        label = torch.tensor(label, dtype=torch.int64)
        return img_transformed, label

class testDataset(data.Dataset):
    '''
    file_list : リスト
        画像パス
    transform: object
        前処理クラスのインスタンス
    phase : 学習化テストか設定する
    '''
    def __init__(self,transform=None, phase='train'):

        self.transform = transform#前処理クラスのリスト
        self.phase = phase#train or val の指定

        rootpath = "./訓練用画像/"
        target_path = osp.join(rootpath+'tests/*.png')
        
        path_list = []

        for path in glob.glob(target_path):
            path_list.append(path)

        self.file_list = path_list

        self.label_dict = {}

        with open('訓練用画像/tests.txt') as f:
            for line in f:
                line = line.replace("\n", "")
                name, label = line.split("/")
                self.label_dict[name] = int(label)
    

    def __len__(self):
        #画像の枚数を返す
        return len(self.file_list)
    
    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルの取得
        '''
        #index番目の画像をロード
        img_path = self.file_list[index]
        file_name = img_path[14:]
        label = self.label_dict[file_name]

        img = Image.open(img_path)
        t_data = Image.new("RGB", img.size, (255, 255, 255))
        t_data.paste(img, mask=img.split()[3])
        if t_data.mode == "RGB":
            t_data = t_data.convert("L")
        #画像の前処理実施
        img_transformed = self.transform(
            t_data, self.phase
        )

        label = torch.tensor(label, dtype=torch.int64)
        return img_transformed, label



class make_datapath_list():

    def _make_datapath_list(ph):
        """
        データパスを格納したリストを作成する
        """

        rootpath = "./訓練用画像/" + ph + "/"
        target_path = osp.join(rootpath+'**/*.png')
    
        path_list = []

        for path in glob.glob(target_path):
            path_list.append(path)

        return path_list
    