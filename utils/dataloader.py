import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

# 假设你的 utils 和 utils_aug 在同一个目录
try:
    from .utils import cvtColor, preprocess_input
    from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
except ImportError:
    print("Warning: Could not import from .utils. Assuming utils.py and utils_aug.py are in the same folder.")
    # Fallback for testing if .utils fails
    from utils import cvtColor, preprocess_input
    from utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


class SiameseDataset(Dataset):
    def __init__(self, input_shape, data_dir, random_flag=True, autoaugment_flag=True):
        self.input_shape = input_shape
        self.data_dir = data_dir
        self.random = random_flag
        self.autoaugment_flag = autoaugment_flag
        
        # 你的4分类任务中，"Not-Lesion" (背景) 的类别 ID
        # (0, 1, 2 是你的病灶类)
        self.negative_class_id = 3 

        # 加载 CC 和 MLO 的正样本，并按 "病人ID_侧别" 建立索引
        print("[Init] Indexing positive samples...")
        self.cc_pos = self._index_positive_images(os.path.join(data_dir, 'CC', 'positive_patches'))
        self.mlo_pos = self._index_positive_images(os.path.join(data_dir, 'MLO', 'positive_patches'))
        
        # 加载负样本池
        print("[Init] Indexing negative samples...")
        self.cc_neg = self._index_negative_images(os.path.join(data_dir, 'CC', 'negative_patches'))
        self.mlo_neg = self._index_negative_images(os.path.join(data_dir, 'MLO', 'negative_patches'))

        # 找到同时拥有 CC 和 MLO 真病灶的病人
        self.valid_patient_ids = list(set(self.cc_pos.keys()) & set(self.mlo_pos.keys()))
        
        # 为了让 epoch 更有意义，我们让其长度等于 (有效病人 * 10)
        # 你也可以设为固定的大数字，例如 10000
        self._len = len(self.valid_patient_ids) * 10 
        
        print(f"[Init] Dataset loaded from: {data_dir}")
        print(f"[Init] Valid paired patients (Patient_Side): {len(self.valid_patient_ids)}")
        print(f"[Init] Positive patches - CC: {len([p for v in self.cc_pos.values() for p in v])}, MLO: {len([p for v in self.mlo_pos.values() for p in v])}")
        print(f"[Init] Negative pool size - CC: {len(self.cc_neg)}, MLO: {len(self.mlo_neg)}")
        print(f"[Init] Epoch size set to: {self._len}")

        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()
            self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        else:
            # 确保即使 autoaugment_flag=False 也有 get_random_data
            pass 

    def _index_positive_images(self, folder):
        """
        索引 positive_patches 文件夹
        文件名格式: P001_L_CC_gtidx0_cls0.jpg
        """
        # idx 结构: {patient_side_id: [(path, cls_id), (path, cls_id), ...]}
        idx = {}
        if not os.path.exists(folder): return idx
        
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                # 文件名示例: P001_L_CC_gtidx0_cls0.jpg
                parts = f.split('_')
                if len(parts) >= 3: # 确保至少有 P001_L_CC
                    # P001_L
                    patient_side_id = f"{parts[0]}_{parts[1]}"
                    
                    # [关键] 解析类别 ID: 找到以 'cls' 开头的部分
                    cls_id = -1 # 默认为 -1 (错误)
                    for part in parts:
                        if part.startswith('cls'):
                            try:
                                # e.g., 'cls0.jpg' -> 0
                                cls_id = int(part.replace('.jpg','').replace('.png','').replace('.jpeg','')[3:])
                            except ValueError:
                                pass 
                            break
                    
                    if cls_id == -1:
                        # print(f"Warning: Could not parse class ID from positive sample: {f}")
                        continue
                        
                    if patient_side_id not in idx: 
                        idx[patient_side_id] = []
                    idx[patient_side_id].append((os.path.join(folder, f), cls_id))
        return idx

    def _index_negative_images(self, folder):
        """
        索引 negative_patches 文件夹
        文件名格式: P003_L_CC_idx0_gtNone_pred0.jpg
        [关键修正]: 所有来自此文件夹的样本，类别ID都是 self.negative_class_id
        """
        # images 结构: [(path, cls_id), (path, cls_id), ...]
        images = []
        if not os.path.exists(folder): return images
        
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                # [关键修正]
                images.append((os.path.join(folder, f), self.negative_class_id))
        return images

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # 'index' 在这里是虚拟的，我们每次都随机选
        if not self.valid_patient_ids:
            raise ValueError("No valid patient IDs found. Check positive CC/MLO folders.")
            
        patient_id = random.choice(self.valid_patient_ids)
        
        # --- Pair 1: 正样本对 (Match = 1) ---
        # 确保 patient_id 真的在 cc_pos 和 mlo_pos 中
        if patient_id not in self.cc_pos or not self.cc_pos[patient_id]:
             raise ValueError(f"Patient ID {patient_id} missing from cc_pos.")
        if patient_id not in self.mlo_pos or not self.mlo_pos[patient_id]:
             raise ValueError(f"Patient ID {patient_id} missing from mlo_pos.")

        # cc_pos[patient_id] 返回 (path, cls_id) 元组列表
        img1_pos_path, img1_cls = random.choice(self.cc_pos[patient_id])
        img2_pos_path, img2_cls = random.choice(self.mlo_pos[patient_id])
        
        # [Match=1, CC类别, MLO类别]
        labels_pos = [1, img1_cls, img2_cls] 

        # --- Pair 2: 负样本对 (Match = 0) ---
        # Anchor (img1) 复用正样本对的 CC
        img1_neg_path, img1_cls_neg = img1_pos_path, img1_cls
        
        # 随机选择一个 "hard negative" (背景) 或 "easy negative" (其他病人)
        if random.random() < 0.5 and len(self.mlo_neg) > 0:
            # 1. Hard Negative: 从 MLO 背景池中选
            img2_neg_path, img2_cls_neg = random.choice(self.mlo_neg)
        else:
            # 2. Easy Negative: 选另一个病人的 MLO 病灶
            other_id = random.choice(self.valid_patient_ids)
            while other_id == patient_id and len(self.valid_patient_ids) > 1:
                other_id = random.choice(self.valid_patient_ids)
            
            if not self.mlo_pos[other_id]:
                 # 如果 other_id 碰巧没有 MLO positive, 退回到 hard negative
                 if len(self.mlo_neg) > 0:
                     img2_neg_path, img2_cls_neg = random.choice(self.mlo_neg)
                 else:
                     # 极端情况：没有 hard negative，只能用 anchor 自己
                     img2_neg_path, img2_cls_neg = img1_neg_path, img1_cls_neg
            else:
                 img2_neg_path, img2_cls_neg = random.choice(self.mlo_pos[other_id])

        # [Match=0, CC类别(anchor), MLO类别(negative)]
        labels_neg = [0, img1_cls_neg, img2_cls_neg]

        return self._load_and_process_batch(
            [img1_pos_path, img2_pos_path], labels_pos,
            [img1_neg_path, img2_neg_path], labels_neg
        )
    
    def _load_and_process_batch(self, pair1_paths, pair1_labels, pair2_paths, pair2_labels):
        # 初始化数组: 2对样本
        # (2, 3, H, W)
        pairs_of_images = [np.zeros((2, 3, self.input_shape[0], self.input_shape[1]), dtype=np.float32) for _ in range(2)]
        
        # [关键修正] 为不同 loss 初始化不同类型的标签
        # (2, 1)
        match_labels = np.zeros((2, 1), dtype=np.float32) # For BCEWithLogitsLoss
        cls1_labels  = np.zeros((2, 1), dtype=np.int64)   # For CrossEntropyLoss
        cls2_labels  = np.zeros((2, 1), dtype=np.int64)   # For CrossEntropyLoss

        # 处理 Pair 1 (Positive)
        self._process_one_pair(0, pair1_paths, pair1_labels, pairs_of_images, match_labels, cls1_labels, cls2_labels)
        # 处理 Pair 2 (Negative)
        self._process_one_pair(1, pair2_paths, pair2_labels, pairs_of_images, match_labels, cls1_labels, cls2_labels)

        return pairs_of_images, (match_labels, cls1_labels, cls2_labels)

    def _process_one_pair(self, idx, paths, labels, images_arr, match_arr, cls1_arr, cls2_arr):
        # 读取和预处理左图 (CC)
        img1 = self._read_and_augment(paths[0])
        images_arr[0][idx, :, :, :] = img1
        # 读取和预处理右图 (MLO)
        img2 = self._read_and_augment(paths[1])
        images_arr[1][idx, :, :, :] = img2
        
        # 赋值标签
        match_arr[idx] = labels[0]
        cls1_arr[idx]  = labels[1]
        cls2_arr[idx]  = labels[2]

    def _read_and_augment(self, path):
        """
        [关键修正]
        1. 确保图像是 3 通道 (RGB)
        2. "选择 A": 根据 'positive'/'negative' 应用不同增强
        """
        try:
            image = Image.open(path)
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            image = Image.new('L', (self.input_shape[0], self.input_shape[1]), color=128) # 返回灰色图像

        # [关键修正 1] 确保 3 通道，VGG 预训练需要
        image = image.convert("RGB")
        
        if self.autoaugment_flag:
            # [关键修正 2] "选择 A" 的逻辑
            if 'positive_patches' in path:
                # 强增强: RandomResizedCrop + Policy
                image = self.AutoAugment(image, random=True)
            else:
                # 弱增强: Resize + CenterCrop
                image = self.AutoAugment(image, random=False)
        else:
            # Fallback: 使用旧的 get_random_data
            image = self.get_random_data(image, self.input_shape, random=self.random)
            
        image = preprocess_input(np.array(image, dtype=np.float32))
        return np.transpose(image, [2, 0, 1])
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
        
    def AutoAugment(self, image, random=True):
        if not random:
            # [关键修正 2] 负样本 (random=False) 的路径
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        # [关键修正 2] 正样本 (random=True) 的路径
        image = self.resize_crop(image) # RandomResizedCrop
        
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image = self.policy(image) # ImageNetPolicy
        return image
    
    # --- get_random_data (旧的增强) 保持不变，作为 fallback ---
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        iw, ih  = image.size
        h, w    = input_shape
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            return image_data
            
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data    = np.array(image, np.uint8)
        r             = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype         = image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data


# ---------------------------------------------
# [关键修正] Collate 函数，处理 LongTensor
# ---------------------------------------------
def dataset_collate(batch):
    left_images, right_images = [], []
    match_labels, cls1_labels, cls2_labels = [], [], []
    
    for pair_imgs, pair_lbls in batch:
        # pair_imgs: [left_batch(2,3,H,W), right_batch(2,3,H,W)]
        # pair_lbls: (match_batch(2,1), cls1_batch(2,1), cls2_batch(2,1))
        
        # 每个 item 有 2 对 (1 正, 1 负)
        for i in range(2): 
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            
            match_labels.append(pair_lbls[0][i])
            cls1_labels.append(pair_lbls[1][i])
            cls2_labels.append(pair_lbls[2][i])

    # images[0] = 所有 CC (left) 图像 (B, 3, H, W)
    # images[1] = 所有 MLO (right) 图像 (B, 3, H, W)
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    
    # [关键修正]
    # match_labels 用于 BCEWithLogitsLoss, 需要 Float
    match_labels_tensor = torch.from_numpy(np.array(match_labels)).type(torch.FloatTensor)
    
    # cls_labels 用于 CrossEntropyLoss, 需要 Long 
    # 并且 .squeeze() 将 (B, 1) 变为 (B,)
    cls1_labels_tensor = torch.from_numpy(np.array(cls1_labels)).type(torch.LongTensor).squeeze()
    cls2_labels_tensor = torch.from_numpy(np.array(cls2_labels)).type(torch.LongTensor).squeeze()
    
    # 返回三个标签 tensor 的元组
    return images, (match_labels_tensor, cls1_labels_tensor, cls2_labels_tensor)

'''
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize

class SiameseDataset(Dataset):
    def __init__(self, input_shape, data_dir, random_flag=True, autoaugment_flag=True):
        self.input_shape = input_shape
        self.data_dir = data_dir
        self.random = random_flag
        self.autoaugment_flag = autoaugment_flag

        # 加载 CC 和 MLO 的正样本，并按 "病人ID_侧别" 建立索引
        self.cc_pos = self._index_images(os.path.join(data_dir, 'CC', 'positive'))
        self.mlo_pos = self._index_images(os.path.join(data_dir, 'MLO', 'positive'))
        
        # 加载负样本池
        self.cc_neg = self._list_images(os.path.join(data_dir, 'CC', 'negative'))
        self.mlo_neg = self._list_images(os.path.join(data_dir, 'MLO', 'negative'))

        # 找到同时拥有 CC 和 MLO 真病灶的病人
        self.valid_patient_ids = list(set(self.cc_pos.keys()) & set(self.mlo_pos.keys()))
        
        print(f"[Init] Dataset loaded from: {data_dir}")
        print(f"[Init] Valid paired samples (Patient_Side): {len(self.valid_patient_ids)}")
        print(f"[Init] Negative pool size - CC: {len(self.cc_neg)}, MLO: {len(self.mlo_neg)}")

        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()
            self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def _index_images(self, folder):
        # idx 结构: {patient_side_id: [(path, cls_id), (path, cls_id), ...]}
        idx = {}
        if not os.path.exists(folder): return idx
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    # 文件名示例: P001_L_CC_cls0_0.jpg
                    parts = f.split('_')
                    if len(parts) >= 4: # 确保有足够的下划线分割部分
                        patient_side_id = f"{parts[0]}_{parts[1]}"
                        
                        # 解析类别 ID: 找到以 'cls' 开头的部分
                        cls_id = 0 # 默认为 0 (例如背景)
                        for part in parts:
                            if part.startswith('cls'):
                                try:
                                    cls_id = int(part[3:]) # 提取 'cls' 后面的数字
                                except ValueError:
                                    pass # 如果解析失败保持默认
                                break
                        
                        if patient_side_id not in idx: 
                            idx[patient_side_id] = []
                        idx[patient_side_id].append((os.path.join(root, f), cls_id))
        return idx

    # --- 修改 2: _list_images 解析类别 ---
    def _list_images(self, folder):
        # images 结构: [(path, cls_id), (path, cls_id), ...]
        images = []
        if not os.path.exists(folder): return images
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    cls_id = 0
                    parts = f.split('_')
                    for part in parts:
                        if part.startswith('cls'):
                            try:
                                cls_id = int(part[3:])
                            except ValueError: pass
                            break
                    images.append((os.path.join(root, f), cls_id))
        return images

    def __len__(self):
        return len(self.valid_patient_ids)

    # --- 修改 3: __getitem__ 使用类别标签 ---
    def __getitem__(self, index):
        patient_id = random.choice(self.valid_patient_ids)
        
        # --- Pair 1: 正样本对 ---
        # 现在 cc_pos[patient_id] 返回的是 (path, cls_id) 元组列表
        img1_pos_path, img1_cls = random.choice(self.cc_pos[patient_id])
        img2_pos_path, img2_cls = random.choice(self.mlo_pos[patient_id])
        # [Match=1, CC类别, MLO类别]
        labels_pos = [1, img1_cls, img2_cls] 

        # --- Pair 2: 负样本对 ---
        img1_neg_path, img1_cls_neg = img1_pos_path, img1_cls # Anchor 复用
        
        if random.random() < 0.5 and len(self.mlo_neg) > 0:
             img2_neg_path, img2_cls_neg = random.choice(self.mlo_neg) # Hard Negative
        else:
             other_id = random.choice(self.valid_patient_ids)
             while other_id == patient_id and len(self.valid_patient_ids) > 1:
                 other_id = random.choice(self.valid_patient_ids)
             img2_neg_path, img2_cls_neg = random.choice(self.mlo_pos[other_id]) # Easy Negative

        labels_neg = [0, img1_cls_neg, img2_cls_neg]

        return self._load_and_process_batch(
            [img1_pos_path, img2_pos_path], labels_pos,
            [img1_neg_path, img2_neg_path], labels_neg
        )


    def _load_and_process_batch(self, path_list):
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros((number_of_pairs, 3, self.input_shape[0], self.input_shape[1])) for _ in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair_idx in range(number_of_pairs):
            image1 = Image.open(path_list[pair_idx * 2])
            image1 = cvtColor(image1)
            if self.autoaugment_flag:
                image1 = self.AutoAugment(image1, random=self.random)
            else:
                image1 = self.get_random_data(image1, self.input_shape, random=self.random)
            image1 = preprocess_input(np.array(image1).astype(np.float32))
            image1 = np.transpose(image1, [2, 0, 1])
            pairs_of_images[0][pair_idx, :, :, :] = image1

            image2 = Image.open(path_list[pair_idx * 2 + 1])
            image2 = cvtColor(image2)
            if self.autoaugment_flag:
                image2 = self.AutoAugment(image2, random=self.random)
            else:
                image2 = self.get_random_data(image2, self.input_shape, random=self.random)
            image2 = preprocess_input(np.array(image2).astype(np.float32))
            image2 = np.transpose(image2, [2, 0, 1])
            pairs_of_images[1][pair_idx, :, :, :] = image2

            labels[pair_idx] = 1 if pair_idx == 0 else 0

        return pairs_of_images, labels
   
# ------------------------------------------------------------------
    def _load_and_process_batch(self, pair1_paths, pair1_labels, pair2_paths, pair2_labels):
        # 初始化数组: 2对样本
        pairs_of_images = [np.zeros((2, 3, self.input_shape[0], self.input_shape[1])) for _ in range(2)]
        # 初始化三个标签数组: [Match, Cls1, Cls2]
        match_labels = np.zeros((2, 1))
        cls1_labels = np.zeros((2, 1))
        cls2_labels = np.zeros((2, 1))

        # 处理 Pair 1
        self._process_one_pair(0, pair1_paths, pair1_labels, pairs_of_images, match_labels, cls1_labels, cls2_labels)
        # 处理 Pair 2
        self._process_one_pair(1, pair2_paths, pair2_labels, pairs_of_images, match_labels, cls1_labels, cls2_labels)

        return pairs_of_images, (match_labels, cls1_labels, cls2_labels)

    def _process_one_pair(self, idx, paths, labels, images_arr, match_arr, cls1_arr, cls2_arr):
        # 读取和预处理左图
        img1 = self._read_and_augment(paths[0])
        images_arr[0][idx, :, :, :] = img1
        # 读取和预处理右图
        img2 = self._read_and_augment(paths[1])
        images_arr[1][idx, :, :, :] = img2
        # 赋值标签
        match_arr[idx] = labels[0]
        cls1_arr[idx] = labels[1]
        cls2_arr[idx] = labels[2]

    def _read_and_augment(self, path):
        image = Image.open(path)
        image = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image = preprocess_input(np.array(image).astype(np.float32))
        return np.transpose(image, [2, 0, 1])
# ----------------------------------------------------------------------------
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        iw, ih  = image.size
        h, w    = input_shape
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            return image_data
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 
        image_data      = np.array(image, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
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
        image = self.resize_crop(image)
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = self.policy(image)
        return image

# ---------------------------------------------
def dataset_collate(batch):
    left_images, right_images = [], []
    match_labels, cls1_labels, cls2_labels = [], [], []
    
    for pair_imgs, pair_lbls in batch:
        # pair_imgs: [left_batch(2,3,H,W), right_batch(2,3,H,W)]
        # pair_lbls: (match_batch(2,1), cls1_batch(2,1), cls2_batch(2,1))
        for i in range(2): # 每个 item 有 2 对
             left_images.append(pair_imgs[0][i])
             right_images.append(pair_imgs[1][i])
             match_labels.append(pair_lbls[0][i])
             cls1_labels.append(pair_lbls[1][i])
             cls2_labels.append(pair_lbls[2][i])

    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    # 返回三个标签 tensor 的元组
    return images, (
        torch.from_numpy(np.array(match_labels)).type(torch.FloatTensor),
        torch.from_numpy(np.array(cls1_labels)).type(torch.FloatTensor),
        torch.from_numpy(np.array(cls2_labels)).type(torch.FloatTensor)
    )
# ----------------------------------------------------    

def dataset_collate(batch):
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

#        def __getitem__(self, index):
#        batch_images_path = []
#    
#        # ------------------------------
#        # 随机选一个类别 c (例如 positive 或 negative)
#        # ------------------------------
#        c = random.randint(0, self.types - 1)
#        same_class_paths = np.array(self.train_lines)[self.train_labels == c]
#    
#        # 按视角拆分
#        cc_paths = [p for p in same_class_paths if 'CC' in p]
#        mlo_paths = [p for p in same_class_paths if 'MLO' in p]
#    
#        # ------------------------------
#        # 从两个不同视角中各选一张 (相似对)
#        # ------------------------------
#        if len(cc_paths) > 0 and len(mlo_paths) > 0:
#            pos_path_1 = random.choice(cc_paths)
#            pos_path_2 = random.choice(mlo_paths)
#        else:
#            # 若某类缺少一个视角，则退化为同视角采样
#            pos_path_1, pos_path_2 = random.sample(list(same_class_paths), 2)
#    
#        batch_images_path.append(pos_path_1)
#        batch_images_path.append(pos_path_2)
#    
#        # ------------------------------
#        # 选出不同类别的一对 (不相似对)
#        # ------------------------------
#        diff_classes = list(range(self.types))
#        diff_classes.remove(c)
#        diff_c = random.choice(diff_classes)
#    
#        diff_class_paths = np.array(self.train_lines)[self.train_labels == diff_c]
#        # 尽量保证来自不同视角
#        diff_cc = [p for p in diff_class_paths if 'CC' in p]
#        diff_mlo = [p for p in diff_class_paths if 'MLO' in p]
#    
#        if len(diff_cc) > 0 and len(diff_mlo) > 0:
#            neg_path_1 = random.choice(diff_cc)
#            neg_path_2 = random.choice(diff_mlo)
#        else:
#            neg_path_1, neg_path_2 = random.sample(list(diff_class_paths), 2)
#    
#        batch_images_path.append(neg_path_1)
#        batch_images_path.append(neg_path_2)
#    
#        # ------------------------------
#        # 转为模型输入格式
#        # ------------------------------
#        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
#        return images, labels


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





