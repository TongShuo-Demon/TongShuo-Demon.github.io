---
title: 观看黑马课程简单记录
description: 设计迁移学习、RCNN、YOLO、SSD
categories:
 - 深度学习
tags:
---



# 迁移学习(Transform learniong)

## 整体介绍

-  利用数据、任务或者模型之间的相似性，比如都是分类问题

- 在旧的领域学习过或者训练好的模型

- 应用于新的领域进行训练

  

## 微调(fine tuning)



- 调整模型参数不需要过多的调整
- 调整模型结构，微型调整
- Pre-trained：预训练模型
- fine tuning：微调之后的模型

## 过程

- 确定当前任务B，修改原始模型结构(比如分类问题最后的全连接层)
- 确定B任务的数据大小
  - B任务的数据量大，放开A模型的所有训练，A结构+修改的全连接层一起训练(A模型有已经训练好的参数)
  - B任务数据量小，将A模型冻掉不去训练，只训练全连接层

# 目标检测概述

## rcnn

图片上选出候选区域，使用SVM进行分类，分类输出，回归得到坐标位置，回归预测采用MSE均方差损失，分类概率采用交叉熵损失。

存在问题：对于多个目标，就不能输出固定个数的位置坐标。

### 步骤：

- 一张图片找到2000个候选区域‘
- 2000个候选区域作大小变换，输入到Alexnet当中，得到特征向量为2000*4096
- SVM分类器进行分类，得到[2000,20]
- 非极大抑制，消除一些重叠度高的
- 修正bbox，对回归作回归

### 存在问题

- 训练时间长
- 训练阶段多
- 处理速度慢
- 图片形状变换

## 检测中的评级指标

- iou(交并比)：两个区域的重叠程度
  - IOU交并比：0~1之间的值
  - 位置的考量
- 平均精确率(mean average precision) map
  - 物体检测的分类准确率
  - 定义：多个分类任务的AP的平均值
    - MAP = 所有类别的AP之和/类别的总个数
    - 对于每个类别计算AP值(AUC)



## sppnet

1，先得到一个feature map

2，原图通过SS得到的候选区域直接映射feature map中对应的位置

左上角点： $x^,=[x/S]+1$,  右下角点：$x^,=[x/S]-1$。S就是CNN中所有的strides的乘积。

3，SPP将映射的候选的特征图转成固定大小的特征向量。原图卷积后为$13*13*256$

- 候选区域的特征图转换成固定大小的特征向量
- SPP会将每一个候选区域分成$1*1$、$2*2$、$4*4$三张子图,对每个子图作最大池化
- （16+4+1）*256 =5376
-  空间盒个数：16+4+1=21





## fast-rcnn

改进：

- 提出一个Roi pooling

- 分类使用softmax

- 与SPPnet一致

  - 得到整张图的feature map
  - 将选择算法的结果(ROI)映射到feature map
  
- Roi Pooling

  - 为了减少计算时间并且得到固定长度的向量
  - 使用4*4=16的空间盒数
  - 这是它比SPP快的原因

- 训练比较统一：废弃了svm以及sppnet

  

- 分类损失使用softmax，采用交叉熵损失；回归损失采用平均绝对误差(MA)即L1损失。

  将二者的和加起来。

  ![](https://gitee.com/MineDemon/picGo/raw/master/2020-11-25rcnn%20VS%20sPPnet.png)

  

  

  ## faster-rcnn

- 四个基本步骤统一到一个步骤里面（候选区域生成、特征提取、分类、位置精修）

- 区域生成网络(RPN) + Fast R-cnn

- rpn替代了SS 选择性搜索算法

  - RPN网络用于生成region proposals
  
  - 通过softmax判断anchors属于物体还是背景
  
  - bbox回归修正anchors获得精确的proposals
  
  - 得到默认300个候选区域继续后面的工作
  
- RPN 原理

  - 用$n*n$（默认$3*3=9$）大小窗口去扫描特征图，得到K个候选窗口。
  - 每个特征图的像素对应9个窗口的大小
  - 三种尺度（128、256、512），三种长宽比（1：1，1：2，2：1）
  - $3*3=9$不同大小的候选框
    - 窗口输出[N,256]--》分类：判断是否是背景
    - 回归位置：N个候选区框与自己对应目标值GT作回归，修正位置
    - 得到更好的候选区域



- 训练
  - rpn训练：
    - 分类：二分类，softmax，logisticsregression
    - 候选框的调整：均方差做修正
  - fast rcnn部分的训练
    - 预测类别训练：softmax
    - 预测位置的训练：均方误差损失
  - 样本准备：正负anchors的比例是1：3





## YOLO

- googleNet + 4个卷积层+2个全连接层

- 网络输出大小：$7*7*30$

### 流程理解

-  单元格(grid cell)
  - $7*7=49$个像素值，理解成49个单元格
  - 每个单元格负责预测一个物体类别，并且直接预测物体的概率值
  - 每个单元格：两个bbox，两个置信度
    - 一个bbox：xmin，ymin，xmax，ymax，confidence
    - 两个bbox：10个值
    - 30：10+20（类别）   

- 网格输出筛选
  - 一个网格预测两个bbox，训练时候只有一个bbox专门预测概率
  - 20个类别概率代表这个网格中的一个bbox
  - 一个confidence score
    - 如果grid cell 里面没有object，confidence就是0
    - 如果有，则confidence score等于预测的box和ground truth的iou乘积
      - 两个bbox与GT进行iou，得到两个iou值
  - yolo框，概率值都直接由网络输出$7*7*30$



### 训练

-  损失 bbox+confidence损失+classification损失



![](https://gitee.com/MineDemon/picGo/raw/master/2020-11-26YOLO.png)



- 缺点：准确率低，对靠近物体检测效果差





## SSD(Single Shot MultiBox Detector)

![](https://gitee.com/MineDemon/picGo/raw/master/2020-11-26SSD.png)

- SSD结合了YOLO的回归思想和Faster-rCNN中的anchor机制

- 不同尺度的特征图上采用卷积核来预测一系列Default Bounding Boxes的类别、坐标偏移
- 不同尺度feature map 所有特点上使用PriorBox层(Detector & classifier)



### Detector & classifier

- PriorBox层：得到default boxes。默认候选框
  - 候选框生成结果后
  - 做微调，利用四个variance
- 卷积：生成localization，4个位置偏移量
- 卷积$3*3$:confidence,21个类别置信度(区分背景)

![](https://gitee.com/MineDemon/picGo/raw/master/2020-11-26SSD_detector.png)

### 训练与测试流程

- 训练-
  - 样本标记，8732个候选区域得到正负样本
  - 正负比例：1：3
  - softmax loss（Faster R-CNN是log loss），位置回归采用的是smooth L1 loss（Faster R-CNN一样）









