from __future__ import print_function

import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import Dataset
import Network



import random
seed = 42
random.seed(seed)  
np.random.seed(seed)  
# PyTorch のRNGを初期化  
torch.manual_seed(seed)

import tqdm

def train_model(net, dataloders_dict, criterion, optimizer, num_epochs):
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    #初期設定
    #GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス", device)
    #モデルをGPUへ
    net.to(device)
    #ネットワークがある程度固定であれば高速化させる
    torch.backends.cudnn.benchmark = True
    #epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-----------------------------------')

        #epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() #モデルを訓練モードに
            else:
                net.eval() #モデルを検証モードに
            epoch_loss = 0.0 #epochの損失0
            epoch_corrects = 0 #epochの正解数
            #データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm.tqdm(dataloders_dict[phase]):
                #optimizerを初期化
                optimizer.zero_grad()
                #順伝搬(forward)計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs,  labels.to("cuda"))#損失を計算
                    _, preds = torch.max(outputs, 1) #ラベルを予測
                    #訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #イテレーション結果の計算
                    #lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    #正解の合計数を更新
                    epoch_corrects += torch.sum(preds.cpu() == labels.data)
            #epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloders_dict[phase].dataset)
            print('{} Loss:{:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
            if phase == 'train':
                train_acc_list.append(epoch_acc)
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc)
                val_loss_list.append(epoch_loss)
    return val_loss_list,train_loss_list, val_acc_list, train_acc_list

def main():

    train_list = Dataset.make_datapath_list._make_datapath_list("tranings")

    size = 28

    #dataLoaderを作成
    train_dataset = Dataset.MyDataset(file_list = train_list, transform=Dataset.ImageTransform(size),phase='train')
    test_dataset = Dataset.testDataset(transform= Dataset.ImageTransform(size), phase='val')

    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)
    test_dataloder = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle=False)    

    #辞書型変数にまとめる
    dataloders_dict = {"train": train_dataloder, "val": test_dataloder}

    #損失関数を設定
    criterion = nn.CrossEntropyLoss()

    model = Network.Net(10)
    model = model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    val_loss_list,train_loss_list, val_acc_list, train_acc_list = train_model(model, dataloders_dict, criterion, optimizer, num_epochs=30)
    torch.save(model.state_dict(), "mnist_cnn.pth")

if __name__ == '__main__':
    main()