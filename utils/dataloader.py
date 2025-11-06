'''
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDataset(Dataset):
    def __init__(self, input_shape, lines, labels, random, autoaugment_flag=True):
        self.input_shape    = input_shape
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)

        self.random         = random
        
        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        batch_images_path = []
        #------------------------------------------#
        #   首先选取三张类别相同的图片
        #------------------------------------------#
        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path)<3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)
        #------------------------------------------#
        #   取出两张类似的图片
        #   对于这两张图片，网络应当输出1
        #------------------------------------------#
        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        #------------------------------------------#
        #   取出两张不类似的图片
        #------------------------------------------#
        batch_images_path.append(selected_path[image_indexes[2]])
        #------------------------------------------#
        #   取出与当前的小类别不同的类
        #------------------------------------------#
        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])
        
        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels

    def _convert_path_list_to_images_and_labels(self, path_list):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)
        #-------------------------------------------#
        #   定义网络的输入图片和标签
        #-------------------------------------------#
        pairs_of_images = [np.zeros((number_of_pairs, 3, self.input_shape[0], self.input_shape[1])) for i in range(2)]
        labels          = np.zeros((number_of_pairs, 1))

        #-------------------------------------------#
        #   对图片对进行循环
        #   0,1为同一种类，2,3为不同种类
        #-------------------------------------------#
        for pair in range(number_of_pairs):
            #-------------------------------------------#
            #   将图片填充到输入1中
            #-------------------------------------------#
            image = Image.open(path_list[pair * 2])
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[0][pair, :, :, :] = image

            #-------------------------------------------#
            #   将图片填充到输入2中
            #-------------------------------------------#
            image = Image.open(path_list[pair * 2 + 1])
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[1][pair, :, :, :] = image
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        #-------------------------------------------#
        #   随机的排列组合
        #-------------------------------------------#
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        return pairs_of_images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image

# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
'''
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

# 保持原仓库的工具导入 (请确保这些文件在你的目录下存在)
from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize

class SiameseDataset(Dataset):
    def __init__(self, input_shape, data_dir, random_flag=True, autoaugment_flag=True):
        """
        Args:
            input_shape: 输入到网络的图像尺寸 (e.g., [105, 105])
            data_dir: 清洗后的数据根目录，内部需包含 CC/positive, CC/negative, MLO/positive, MLO/negative
            random_flag: 是否启用基础随机增强
            autoaugment_flag: 是否启用高级自动增强 (AutoAugment)
        """
        self.input_shape = input_shape
        self.data_dir = data_dir
        self.random = random_flag
        self.autoaugment_flag = autoaugment_flag

        # --- 1. 构建医学数据索引 ---
        # 加载 CC 和 MLO 的正样本，并按 "病人ID_侧别" 建立索引
        self.cc_pos = self._index_images(os.path.join(data_dir, 'CC', 'positive'))
        self.mlo_pos = self._index_images(os.path.join(data_dir, 'MLO', 'positive'))
        
        # 加载负样本池 (背景/假阳性)，用于构建困难负样本对
        self.cc_neg = self._list_images(os.path.join(data_dir, 'CC', 'negative'))
        self.mlo_neg = self._list_images(os.path.join(data_dir, 'MLO', 'negative'))

        # 找到既有 CC 又有 MLO 真病灶的 "合格样本" (Intersection)
        # 只有这些样本才能构成完整的正样本对
        self.valid_patient_ids = list(set(self.cc_pos.keys()) & set(self.mlo_pos.keys()))
        
        # 打印数据集统计信息
        print(f"[Init] Dataset loaded from: {data_dir}")
        print(f"[Init] Valid paired samples (Patient_Side): {len(self.valid_patient_ids)}")
        print(f"[Init] Negative pool size - CC: {len(self.cc_neg)}, MLO: {len(self.mlo_neg)}")

        # --- 初始化图像增强器 (保持原作者逻辑) ---
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()
            self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def _index_images(self, folder):
        idx = {}
        if not os.path.exists(folder): return idx
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                # 文件名: 0a30..._L_CC_4.jpg
                # 使用 '_' 分割后: ['0a30...', 'L', 'CC', '4.jpg']
                parts = f.split('_')
                
                # 取前两部分组合成唯一ID: "0a30..._L"
                patient_side_id = f"{parts[0]}_{parts[1]}"
                
                if patient_side_id not in idx: 
                    idx[patient_side_id] = []
                idx[patient_side_id].append(os.path.join(folder, f))
        return idx

    def _list_images(self, folder):
        """辅助函数：简单列出文件夹下所有图像路径"""
        if not os.path.exists(folder): return []
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        # 定义一个 epoch 的迭代次数。
        # 这里设置为合格样本数量，你也可以乘以一个系数让每个 epoch 跑得更久
        return len(self.valid_patient_ids)

    def __getitem__(self, index):
        # 每次调用返回 2 对样本 (4张图)，以适配原作者的 batch处理逻辑
        # Pair 1: 正样本对 (Label=1) -> [CC真, MLO真] (同一病人同一侧)
        # Pair 2: 负样本对 (Label=0) -> [CC真, MLO假] (或是其他病人的真)

        # 1. 随机选择一个锚点 (Anchor Patient ID)
        # 我们用 random.choice 而不是 index，增加了随机性
        patient_id = random.choice(self.valid_patient_ids)
        
        # --- 构建 Pair 1 (正样本对) ---
        # 从该 ID 下随机选一张 CC 真病灶和一张 MLO 真病灶
        img1_pos = random.choice(self.cc_pos[patient_id])
        img2_pos = random.choice(self.mlo_pos[patient_id])

        # --- 构建 Pair 2 (负样本对) ---
        # Anchor 复用上面的 CC 真病灶
        img1_neg = img1_pos 
        
        # Negative 的选择策略 (混合策略以提升鲁棒性):
        # 50% 概率选择: "困难负样本" (Hard Negative) -> 同一个视角下的假阳性背景
        # 50% 概率选择: "简单负样本" (Easy Negative) -> 另一个病人的真病灶
        if random.random() < 0.5 and len(self.mlo_neg) > 0:
             # 选择 MLO 视角的假阳性背景
             img2_neg = random.choice(self.mlo_neg)
        else:
             # 选择另一个不同病人的 MLO 真病灶
             other_id = random.choice(self.valid_patient_ids)
             # 确保选到的不是同一个人
             while other_id == patient_id and len(self.valid_patient_ids) > 1:
                 other_id = random.choice(self.valid_patient_ids)
             img2_neg = random.choice(self.mlo_pos[other_id])

        # 将 4 张图的路径打包
        # 顺序: [Pair1_图1, Pair1_图2, Pair2_图1, Pair2_图2]
        batch_path_list = [img1_pos, img2_pos, img1_neg, img2_neg]
        
        # 读取图像、预处理并生成标签
        images, labels = self._load_and_process_batch(batch_path_list)
        return images, labels

    def _load_and_process_batch(self, path_list):
        """
        读取路径列表中的图像，进行增强和预处理，并分配标签。
        适配原作者的返回格式: (2, batch, 3, h, w)
        """
        number_of_pairs = int(len(path_list) / 2)
        # 初始化两个空的数组用于存放左右两组图片
        pairs_of_images = [np.zeros((number_of_pairs, 3, self.input_shape[0], self.input_shape[1])) for _ in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair_idx in range(number_of_pairs):
            # --- 处理左图 (Image 1) ---
            image1 = Image.open(path_list[pair_idx * 2])
            image1 = cvtColor(image1)
            if self.autoaugment_flag:
                image1 = self.AutoAugment(image1, random=self.random)
            else:
                image1 = self.get_random_data(image1, self.input_shape, random=self.random)
            # 预处理并转置为 CHW 格式
            image1 = preprocess_input(np.array(image1).astype(np.float32))
            image1 = np.transpose(image1, [2, 0, 1])
            pairs_of_images[0][pair_idx, :, :, :] = image1

            # --- 处理右图 (Image 2) ---
            image2 = Image.open(path_list[pair_idx * 2 + 1])
            image2 = cvtColor(image2)
            if self.autoaugment_flag:
                image2 = self.AutoAugment(image2, random=self.random)
            else:
                image2 = self.get_random_data(image2, self.input_shape, random=self.random)
            image2 = preprocess_input(np.array(image2).astype(np.float32))
            image2 = np.transpose(image2, [2, 0, 1])
            pairs_of_images[1][pair_idx, :, :, :] = image2

            # --- 分配标签 ---
            # 根据我们在 __getitem__ 中的构建顺序:
            # pair_idx=0 是正样本对 (Label=1)
            # pair_idx=1 是负样本对 (Label=0)
            if pair_idx == 0:
                labels[pair_idx] = 1
            else:
                labels[pair_idx] = 0

        return pairs_of_images, labels

    # --- 以下直接保留原作者的图像增强辅助函数 ---
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # (原作者代码，此处省略以节省空间，请确保文件中保留了此函数)
        # ... [请复制原仓库的 get_random_data 实现] ...
        pass # 实际使用时请替换为完整代码

    def AutoAugment(self, image, random=True):
        # (原作者代码，此处省略)
        # ... [请复制原仓库的 AutoAugment 实现] ...
        pass # 实际使用时请替换为完整代码

# --- DataLoader 的 collate_fn (保持原样或微调) ---
def dataset_collate(batch):
    # 将多个 __getitem__ 返回的结果合并为一个大的 batch
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return images, labels
    
