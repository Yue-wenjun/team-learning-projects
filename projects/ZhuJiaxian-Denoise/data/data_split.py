''' 将"./original"进行划分。划分的结果保存于trainset.txt和testset.txt'''

import os, random
os.chdir(os.path.dirname(__file__))
rate = 0.9
names = os.listdir('original/')
amount_of_train = round(rate*len(names))

random.shuffle(names)
train, test = names[:amount_of_train], names[amount_of_train:]
# with open('trainset.txt','w') as file:
#     file.write('\n'.join(train))
# with open('testset.txt','w') as file:
#     file.write('\n'.join(test))

# # 数据压缩: 将全部数据压缩到.npz文件

# import numpy as np
# import cv2
# with open('trainset.txt','r') as file:
#     train = file.read().split('\n')
# with open('testset.txt','r') as file:
#     test = file.read().split('\n')
# X_train = np.zeros((len(train), 160,160,3),np.uint8)
# Y_train = np.zeros((len(train), 160,160,3),np.uint8)
# X_test  = np.zeros((len(test), 160,160,3),np.uint8)
# Y_test  = np.zeros((len(test), 160,160,3),np.uint8)

# for i, name in enumerate(train):
#     # x = cv2.resize(cv2.imread('noisy/'+name), (160,160))
#     # X_train[i,:,:,:] = x
#     y = cv2.imread('original/'+name)
#     Y_train[i,:,:,:] = y

# for i, name in enumerate(test):
#     # x = cv2.resize(cv2.imread('noisy/'+name), (160,160))
#     # X_test[i,:,:,:] = x
#     y = cv2.imread('original/'+name)
#     Y_test[i,:,:,:] = y

# # np.savez('data.npz', x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)
# np.savez('images.npz', train=Y_train, test=Y_test)
