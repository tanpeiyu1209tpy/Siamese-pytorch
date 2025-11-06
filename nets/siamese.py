import torch
import torch.nn as nn

from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        del self.vgg.avgpool
        del self.vgg.classifier
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])

        # matching branch
        #self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect1 = torch.nn.Linear(flat_shape*2, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        
# ------------------------------------------------------
        # classification branch
        self.cls_head_cc = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1) # 输出一个分类 Logit (是病灶/不是病灶)
        )
        # MLO 分支可以独立，也可以和 CC 共享权重。论文似乎是独立的。
        self.cls_head_mlo = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
# --------------------------------------------------------

    def forward(self, x):
        '''
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)   
        #-------------------------#
        #   相减取绝对值，取l1距离
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        #x = torch.abs(x1 - x2)
        x = torch.cat([x1, x2], dim=1)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
        '''
        img1, img2 = x
        
        # fetature extraction
        f1 = self.vgg.features(img1)
        f2 = self.vgg.features(img2)
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)

        # pairing 
        x_match = torch.cat([f1, f2], dim=1) # concat
        x_match = self.match_fc1(x_match)
        match_score = self.match_fc2(x_match)

        # classifying 
        cls_score1 = self.cls_head_cc(f1)   # CC图片分类得分
        cls_score2 = self.cls_head_mlo(f2)  # MLO图片分类得分

        return match_score, cls_score1, cls_score2
