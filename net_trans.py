import torch
import torch.nn as nn
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling


class Cnn_With_Clinical_Net(nn.Module):
    def __init__(self, model):
        super(Cnn_With_Clinical_Net, self).__init__()
        # self.layer = nn.Sequential(*list(model.children())[:-1])
        # self.feature = list(model.children())[-1].in_features
        # self.cnn = nn.Linear(self.feature, 128)

        # CNN
        for i, p in enumerate(model.parameters()):
            if i < 113:
                p.requires_grad = False
        self.layer = nn.Sequential(*list(model.children()))  # 提取模型中的层
        self.conv = self.layer[:-1]  # 提取到倒数第二层
        self.dense = None  # 原始模型的全连接层定义为空，更改模型
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 128)

        # clinical feature
        self.clinical = nn.Linear(100, 100)  # 全连接层输入/输出size（临床数据的特征提取，输入特征为55维向量）

        # concat合并
        self.mcb = CompactBilinearPooling(128, 100, 128).cuda()  # 双线性池化，特征融合输入1/输入2/输出 size
        # self.concat = nn.Linear(128+55, 128)
        self.bn = nn.BatchNorm1d(128)  # 标准化
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x, clinical_features):  # 反向传播（x为图像特征，clinical_features为临床特征）
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 相当于reshape，-1表示不确定
        if self.dense is not None:
            x = self.dense(x)
        x = self.linear(x)
        clinical = self.clinical(clinical_features)
        x = self.mcb(x, clinical)
        # x = torch.cat([x, clinical], dim=1)
        # x = self.concat(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class Net(nn.Module):  # 更改分类数
    def __init__(self, model):
        super(Net, self).__init__()
        for i, p in enumerate(model.parameters()):
            if i < 113:
                p.requires_grad = False
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = self.linear(x)
        return x