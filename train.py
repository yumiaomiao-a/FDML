#_*_coding:utf-8 _*_
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import recall_score,f1_score,precision_score
import cv2
from data_augmentation import face_eraser_change, bg_eraser_change
import random
import argparse


# read data
data_dir = './FF++' # your data dir

data_transform = {
    'train':transforms.Compose([
        transforms.Scale([299,299]),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'test':transforms.Compose([
        transforms.Scale([299,299]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                         transform = data_transform[x]) for x in ['train', 'test']}
train_set = image_datasets['train']
test_set = image_datasets['test']

dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 8,
                                            shuffle = True,drop_last=True,num_workers=4,pin_memory=True) for x in ['train','test'] }  # 读取完数据后，对数据进行装载

train_dataloader = dataloader['train']
test_dataloader = dataloader ['test']

dataset_size = {x:len(image_datasets[x]) for x in ['train','test']}

from model_final_efficient import FDML
model = FDML()
print('----------------',model)

# dic = torch.load('./save_result_eff/ffc23.pth')
# # new_state_dict = {}
# # for k,v in dic.items():
# #     new_state_dict[k[7:]] = v
# # model.load_state_dict(new_state_dict)

def adjust_learning_rate(epoch):
    lr = 0.0002
    if epoch > 10:
        lr = lr / 10
    elif epoch > 20:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 1000
    elif epoch > 40:
        lr = lr / 10000
    elif epoch > 50:
        lr = lr / 100000
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


loss_f = torch.nn.CrossEntropyLoss()
#loss_f = torch.nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002,weight_decay= 4e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
# optimizer = torch.optim.Adadelta(model.parameters(),lr=0.0002)
myscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20)

# multi-gpu
model = torch.nn.DataParallel(model,device_ids=[0,1])
model = model.cuda()

epoch_n = 50


def save_models(epoch):
    torch.save(model.state_dict(), "./result/fs23_{}.pth".format(epoch)) # model save dir
print("Chekcpoint saved")


def train(num_epochs):
    best_acc = 0.0
    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            i+=1
            images = images.cuda()
            labels = labels.cuda()

            # data augmentation
            images_aug = images
            for j in range(len(images)):
                try:
                    l = labels[j]
                    fill = []
                    for m in range(len(images)):
                        if labels[m] == l:
                            fill.append(images[m])
                    p = np.random.rand()
                    if 0<p<0.5:
                        images_aug[j] = face_eraser_change(images[j],fill)
                    else:
                        images_aug[j] = bg_eraser_change(images[j],fill)
                except:
                    images_aug[j] = images[j]

            # data augmentation: Mixup
            # images_aug = images
            # for j in range(len(images)):
            #     try:
            #         l = labels[j]
            #         fill = []
            #         for m in range(len(images)):
            #             if labels[m]==l:
            #                 fill.append(images[m])
            #
            #         if l==0:
            #             images_aug[j] = fake_mix_fake(images[j],fill)
            #         else:
            #             images_aug[j] = real_mix_real(images[j],fill)
            #
            #     except:
            #         images_aug[j] = images[j]


            s1,s2,s3,s4, y1, y2, z1, z2= model(images,images_aug)

            loss1 = loss_f(s2, labels) # s2 means forgery-relevant feature
            labels_ver = torch.ones(len(images))
            labels_ver = torch.tensor(labels_ver,dtype=torch.int64)
            labels_ver = labels_ver.cuda()
            loss2 = loss_f(s1, labels_ver) # s1 means forgery-irrelevant feature, for real image, s1 is real
            loss3 = loss_f(s4, labels)
            loss4 = loss_f(s3, labels_ver)

            loss = loss1 + loss2 + loss3 + loss4
            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            # _, prediction = torch.max(outputs.data, 1)
            s = (s2 + s4) / 2
            _, prediction2 = torch.max(s.data, 1)
            prediction = prediction2

            train_acc += torch.sum(prediction == labels.data)
            batch_size,m,mm,mmm = images.shape

            if i % 5 == 0:
                batch_loss = train_loss / (batch_size*i)
                batch_acc = train_acc / (batch_size*i)
                print(
                    'Epoch[{}] batch[{}],Loss:{:.4f},Acc:{:.4f}'.format(
                        epoch, i, batch_loss, batch_acc))

            torch.cuda.empty_cache()
            myscheduler.step()

        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)

        print(
            "Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(
                epoch, train_acc, train_loss))


if __name__ == '__main__':
    train(50)


