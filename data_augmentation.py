#_*_coding:utf-8 _*_
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
import cv2


def face_eraser_change(input_img,dir):
    dir = torch.tensor([item.gpu().detach().numpy() for item in dir]).cuda()
    dirlen = len(dir)
    i = random.randint(0,dirlen-1)
    fillimage = dir[i]

    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    input_img = input_img.gpu().numpy()*255
    input_img = input_img.astype('uint8')
    input_img = np.transpose(input_img,(1,2,0))
    fillimage = fillimage.gpu().numpy()*255
    fillimage = fillimage.astype('uint8')
    fillimage = np.transpose(fillimage,(1,2,0))

    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2BGRA)
    face = face_detector.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=3,minSize=(32,32), flags=4)

    for (x,y,w,h) in face:
        w = int(w)
        h = int(h)
        # print("%%%%%%%%%%",w,h)
        # face_new = get_one_image(dir)
        face_new = fillimage
        # resize = Resize([w,h])
        # face_new = resize(face_new)
        face_new = cv2.resize(face_new,dsize=(w,h))
        # face_new = cv2.cvtColor(np.asarray(face_new),cv2.COLOR_RGB2BGR)
        face_new = Image.fromarray(face_new)
        input_img = Image.fromarray(input_img)
        # print("_________",face_new.shape)
        # print("_________",face_new.dtype)
        input_img.paste(face_new,(x,y))
        #print("+++++++++__________",type(input_img))
        # input_img = input_img(Rect(ymax,xmax))
        face_eraser_change = cv2.cvtColor(np.asarray(input_img),cv2.COLOR_RGB2BGR)
        face_eraser_change = face_eraser_change.astype('float32')
        face_eraser_change = torch.from_numpy(face_eraser_change)
        face_eraser_change = np.transpose(face_eraser_change,(2,0,1))
        return face_eraser_change


def bg_eraser_change(input_img,dir):
    dir = torch.tensor([item.gpu().detach().numpy() for item in dir]).cuda()
    dirlen = len(dir)
    i = random.randint(0,dirlen-1)
    fillimage = dir[i]

    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    input_img = input_img.gpu().numpy()*255
    input_img = input_img.astype('uint8')
    input_img = np.transpose(input_img,(1,2,0))
    fillimage = fillimage.gpu().numpy()*255
    fillimage = fillimage.astype('uint8')
    fillimage = np.transpose(fillimage,(1,2,0))

    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2BGRA)
    face = face_detector.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=3,minSize=(32,32), flags=4)

    for (xmax,ymax,w,h) in face:
        w = int(w)
        h = int(h)
        faceimg = input_img[ymax:ymax+h,xmax:xmax+w]
        #faceimg = cv2.cvtColor(np.asarray(faceimg),cv2.COLOR_RGB2BGR)
        faceimg = Image.fromarray(faceimg)
        #print("org  face width = ",m1)
        bg_new = fillimage
        bg_gray = cv2.cvtColor(np.asarray(bg_new),cv2.COLOR_COLOR_BGR2BGRA)
        facenew = face_detector.detectMultiScale(bg_gray, scaleFactor=1.05,minNeighbors=3,minSize=(32,32), flags=4)
        for (a,b,c,d) in facenew:
            c = int(c)
            d = int(d)
            faceimg = cv2.resize(faceimg,dsize=(c,d))
            faceimg = Image.fromarray(faceimg)
            bg_new = Image.fromarray(bg_new)
            bg = faceimg.paste(bg_new,(a,b))
            bg_eraser_change = cv2.cvtColor(np.asarray(bg),cv2.COLOR_RGB2BGR)
            bg_eraser_change = bg_eraser_change.astype('float32')
            bg_eraser_change = torch.from_numpy(bg_eraser_change)
            bg_eraser_change = np.transpose(bg_eraser_change,(2,0,1))
            return bg_eraser_change


def get_one_image(dir):
    files = os.listdir(dir)
    n = len(files)
    ind = np.random.randint(0,n)
    file_dir = os.path.join(dir,files[ind])
    subfiles = os.listdir(file_dir)
    m = len(subfiles)

    if m == 1:
        ind1 = int(0)
        img = os.path.join(file_dir,subfiles[ind1])
        image = Image.open(img)

    elif m == 0:
        os.removedirs(file_dir)

    else:
        ind1 = np.random.randint(0,m)
        img = os.path.join(file_dir,subfiles[ind1])
        image = Image.open(img)

    return image



# if __name__ == '__main__':
#
#     import cv2
#
#     face_detector = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
#
#     frame = cv2.imread('real.jpg')
#
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
#     print('++++++',gray.dtype)
#
#     # print('######',frame)
#     # frame = cv2.imread(img)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_detector.detectMultiScale(frame, 1.3, 3)
#     # print('+++++++',faces.shape)
#     print('+++++++',faces.dtype)
#     # print('+++++++',type(faces))
#
#
#
#     for face in faces:
#         cv2.rectangle(frame, (face[0], face[1]), (face[0]+face[2],face[1]+face[3]), (0, 0, 255), 2)
#     cv2.imshow("test", frame)
#     cv2.waitKey(0)
#
#
#
#
#
#     img = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img = img[...,::-1]
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img)
#     plt.show()
#
#
#
#
#
#
#     imgg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img1 = face_eraser_gray(imgg)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img1)
#     plt.show()
#
#     imgggg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img3 = face_eraser_shuffle(imgggg)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img3)
#     plt.show()
#
#     path = '/home/zyq/Desktop/second paper/dataset/FF++/manipulated_sequences/Deepfakes/c40/df_60face(1.3)/'
#     imgggggg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img5 = face_eraser_change(imgggggg,path)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img5)
#     plt.show()
#
#
#     #fig = plt.figure()
#     #ax1 = fig.add_subplot(141)
#     #ax1 = plt.imshow(img)
#
#     #ax2 = fig.add_subplot(142)
#     #ax2 = plt.imshow(img1)
#
#     #ax3 = fig.add_subplot(143)
#     #ax3 = plt.imshow(img3)
#
#     #ax4 = fig.add_subplot(144)
#     #ax4 = plt.imshow(img5)
#
#     #plt.show()
#
#
#
#
# #factuals
#     img = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     print('______-',type(img))
#     print('______-',img.shape)
#
#
#
#     img = img[...,::-1]
#
#     imggg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img2 = bg_eraser_gray(imggg)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img2)
#     plt.show()
#
#     imggggg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     img4 = bg_eraser_shuffle(imggggg)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img4)
#     plt.show()
#
#     imggggggg = cv2.imread('./celeb-df-120-60(1.3)/test/real/0098/00000.jpg')
#     path = '/home/zyq/Desktop/second paper/dataset/FF++/manipulated_sequences/Deepfakes/c40/df_face_clip/'
#     img6 = bg_eraser_change(imggggggg,path)
#     plt.xticks([]) # 不显示x轴
#     plt.yticks([]) # 不显示y轴
#     plt.imshow(img6)
#     plt.show()
#
#     #fig = plt.figure()
#     #ax5 = fig.add_subplot(141)
#     #ax5 = plt.imshow(img)
#
#     #ax6 = fig.add_subplot(142)
#     #ax6 = plt.imshow(img2)
#
#     #ax7 = fig.add_subplot(143)
#     #ax7 = plt.imshow(img4)
#
#     #ax8 = fig.add_subplot(144)
#     #ax8 = plt.imshow(img6)
#
#     #plt.show()


