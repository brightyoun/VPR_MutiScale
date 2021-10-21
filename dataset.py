import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from PIL import Image

import settings 


class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        # self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
        # self.file_num = len(self.mat_files)

        self.x_dir = self.root_dir + '\\input\\'
        self.y_dir = self.root_dir + '\\ground_truth\\'
        self.matx_files = os.listdir(self.x_dir)
        self.maty_files = os.listdir(self.y_dir)
        self.filex_num = len(self.matx_files)
        self.filey_num = len(self.maty_files)
        print("김예찬3 : ", self.filex_num, self.filey_num)

    def __len__(self):
        return self.filex_num

    def __getitem__(self, idx):

        ''' 1. Train'''
        print("트레인 확인 : ", self.x_dir, self.y_dir)
        print("111- idx는 ?  : ", idx)
        xfile_name = self.matx_files[idx % self.filex_num]
        print("222- idx는 ?  : ", idx)
        yfile_name = self.maty_files[idx % self.filey_num]
        print("파일네임11111 : ", xfile_name, self.filex_num, yfile_name)
        x_label = np.array(idx)
        print("파일네임2222222222222: ", x_label)

        ximg_file = os.path.join(self.x_dir, xfile_name)
        yimg_file = os.path.join(self.y_dir, yfile_name)

        ximg_pair = cv2.imread(ximg_file).astype(np.float32) / 255
        yimg_pair = cv2.imread(yimg_file).astype(np.float32) / 255

        ximg_pair = cv2.resize(ximg_pair, dsize=(120, 80), interpolation=cv2.INTER_AREA)
        yimg_pair = cv2.resize(yimg_pair, dsize=(120, 80), interpolation=cv2.INTER_AREA)

        O = ximg_pair #더러운거
        B = yimg_pair # 백그라운드
        L = x_label
        # cv2.imshow('Test', O)
        # cv2.waitKey(-1)
        # cv2.imshow('Test2', B)
        # cv2.waitKey(-1)



        ''' Original'''
        # file_name = self.mat_files[idx % self.file_num]
        # print("파일네임 : ", file_name, self.__len__(), idx)
        # img_file = os.path.join(self.root_dir, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255
        # img_pair = cv2.resize(img_pair, dsize=(480,320),interpolation=cv2.INTER_AREA)
        # print("오픈씨브이 테스트 : ", img_pair.shape)
        #
        # if settings.aug_data:
        #     O, B = self.crop(img_pair, aug=True)
        #     O, B = self.flip(O, B)
        #     O, B = self.rotate(O, B)
        # else:
        #     O, B = self.crop(img_pair, aug=False)

        print("점검용", type(O), O.shape, B.shape)
        # ooo = cv2.imread(O)
        # cv2.imshow('Test', ooo)
        # cv2.waitKey(-1)

        cv2.imwrite("filename.png", O)
        cv2.imwrite("filename22.png", B)
        # im = Image.fromarray(O)
        # im.save("your_file.jpeg")


        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B, 'L' : L}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        # cv2.imshow('Test', O)
        # cv2.waitKey(-1)
        # cv2.imshow('Test2', B)
        # cv2.waitKey(-1)

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))

        print("로테이트 테스트", O.shape, B.shape, type(O))
        #oooo = cv2.imdecode(np.asarray(bytearray(O)), cv2.IMREAD_COLOR)
        #oooo = np.array(O, np.uint8)
        print(O)
        # oooo = cv2.imread(O, 1)
        # #print("111 : ", np.shape(oooo))
        # #bbbb = cv2.imdecode(np.asarray(bytearray(B)), cv2.IMREAD_COLOR)
        # #B = np.array(B, np.uint8)
        # bbbb = cv2.imread(B, 1)

        # cv2.imshow('Test', O)
        # cv2.waitKey(-1)
        # cv2.imshow('Test2', B)
        # cv2.waitKey(-1)

        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        # self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
        # self.file_num = len(self.mat_files)

        self.x_dir = self.root_dir + '\\input\\'
        self.y_dir = self.root_dir + '\\ground_truth\\'
        self.matx_files = os.listdir(self.x_dir)
        self.maty_files = os.listdir(self.y_dir)
        self.filex_num = len(self.matx_files)
        self.filey_num = len(self.maty_files)
        print("김예찬3 : ", self.filex_num, self.filey_num)

    def __len__(self):
        return self.filex_num

    def __getitem__(self, idx):

        ''' 1. Train'''
        print("트레인 확인 : ", self.x_dir, self.y_dir)
        print("111- idx는 ?  : ", idx)
        xfile_name = self.matx_files[idx % self.filex_num]
        print("222- idx는 ?  : ", idx)
        yfile_name = self.maty_files[idx % self.filey_num]
        print("파일네임11111 : ", xfile_name, self.filex_num, yfile_name)

        ximg_file = os.path.join(self.x_dir, xfile_name)
        yimg_file = os.path.join(self.y_dir, yfile_name)

        ximg_pair = cv2.imread(ximg_file).astype(np.float32) / 255
        yimg_pair = cv2.imread(yimg_file).astype(np.float32) / 255

        ximg_pair = cv2.resize(ximg_pair, dsize=(480, 320), interpolation=cv2.INTER_AREA)
        yimg_pair = cv2.resize(yimg_pair, dsize=(480, 320), interpolation=cv2.INTER_AREA)

        O = ximg_pair #더러운거
        B = yimg_pair # 백그라운드
        cv2.imshow('Test', O)
        cv2.waitKey(-1)
        cv2.imshow('Test2', B)
        cv2.waitKey(-1)



        ''' Original'''
        # file_name = self.mat_files[idx % self.file_num]
        # print("파일네임 : ", file_name, self.__len__(), idx)
        # img_file = os.path.join(self.root_dir, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255
        # img_pair = cv2.resize(img_pair, dsize=(480,320),interpolation=cv2.INTER_AREA)
        # print("오픈씨브이 테스트 : ", img_pair.shape)
        #
        # if settings.aug_data:
        #     O, B = self.crop(img_pair, aug=True)
        #     O, B = self.flip(O, B)
        #     O, B = self.rotate(O, B)
        # else:
        #     O, B = self.crop(img_pair, aug=False)

        print("점검용", type(O), O.shape, B.shape)
        # ooo = cv2.imread(O)
        # cv2.imshow('Test', ooo)
        # cv2.waitKey(-1)

        cv2.imwrite("filename.png", O)
        cv2.imwrite("filename22.png", B)
        # im = Image.fromarray(O)
        # im.save("your_file.jpeg")


        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        #h_8 = h % 8
        #w_8 = w % 8
        if settings.pic_is_pair:
            O = np.transpose(img_pair[:, w:], (2, 0, 1))
            B = np.transpose(img_pair[:, :w], (2, 0, 1))
        else:
            O = np.transpose(img_pair[:, :], (2, 0, 1))
            B = np.transpose(img_pair[:, :], (2, 0, 1))
        sample = {'O': O, 'B': B,'file_name':file_name}

        return sample


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
