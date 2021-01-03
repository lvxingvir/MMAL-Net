import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
import torch
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CUB():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        box_file = open(os.path.join(self.root, 'bounding_boxes.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        box_file_list = []
        for line in box_file:
            data = line[:-1].split(' ')
            box_file_list.append([int(float(data[2])), int(float(data[1])),
                                  int(float(data[4])), int(float(data[3]))])
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if i])
        self.test_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if not i])
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = imageio.imread(self.train_img[index]), self.train_label[index], self.train_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target, box = imageio.imread(self.test_img[index]), self.test_label[index], self.test_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(448)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        scale = torch.tensor([height_scale, width_scale])

        return img, target, box, scale

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class STANFORD_CAR():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'cars_train')
        test_img_path = os.path.join(self.root, 'cars_test')
        train_label_file = open(os.path.join(self.root, 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)


        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class FGVC_aircraft():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class MURA_Dataset(Dataset):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, data_dir, csv_file, transform=None):
        """
        :param data_dir: the directory of data
        :param csv_file: the .csv file of data list
        :param transform: the transforms exptected to be applied to the data
        """
        self.data_dir = data_dir
        ori_df = pd.read_csv(os.path.join(data_dir, csv_file))
        # self.frame = ori_df.sample(frac=0.01,random_state=1).reset_index(drop=True)  # for network testing
        self.frame = ori_df.copy()
        self.transform = transform

    def _parse_patient(self, img_filename):
        """
        :param img_filename: the file name of the image data
        :return: the number of patient
        """
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        """
        :param img_filename: the file name of the image data
        :return: the number of study
        """
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        """
        :param img_filename: the file name of image data
        :return: the number of image
        """
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        """
        :param img_filename: the file name of image data
        :return: the type of the study
        """
        return self._study_type_re.search(img_filename).group(1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_filename = self.frame.loc[idx]['FilePath']
        # print(idx,img_filename)
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        file_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(file_path).convert('RGB')
        label = self.frame.loc[idx]['Label']
        label = int(label)
        label_bp = self.frame.loc[idx]['Bp_label']
        label_bp = int(label_bp)

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'file_path': file_path,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }

        if self.transform:
            image = self.transform(image)

        # plt.imshow(image.permute(1,2,0).numpy())
        # sample = {'image': image, 'label': label, 'meta_data': meta_data}
        # return image,label # for only mura dataset
        return image,label_bp,label   # for mura bodypart