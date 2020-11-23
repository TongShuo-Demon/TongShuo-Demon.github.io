---
title: 基于darknet训练yolo模型以及opencv的工程文件驱动此模型
description: 使用yolov3训练自己的数据集
categories:
 - opencv
 - 深度学习
tags:
---



## 首先安装darknet

在终端中输入如下指令：

```powershell
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```

在cfg子目录中就存在yolo的配置文件了，下一步就要下载权重文件了

```powershell
wget https://pjreddie.com/media/files/yolov3.weights
```

然后运行检测器

```powershell
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
```

输出结果如下所示

```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    .......
  105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
  106 detection
truth_thresh: Using default '1.000000'
Loading weights from yolov3.weights...Done!
data/dog.jpg: Predicted in 0.029329 seconds.
dog: 99%
truck: 93%
bicycle: 99%
```

![undefined](http://ww1.sinaimg.cn/large/006lMPXUgy1gdtnalbq85j31cw15a1kx.jpg)

更改检测阈值

```powershell
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0
```

以上内容可以说都是参考[darknet](https://pjreddie.com/darknet/yolo/)的官网资料

## 训练自己的模型

如果只按照官网走一遍流程那肯定不能玩的痛快啊！所以必须要会训练自己的数据集。

### 安装标注文件软件labelimg

python3+QT5

```powershell
sudo apt-get install pyqt5-dev-tools
sudo pip3 install lxml
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
make all
python3 labelImg.py  #打开labelImg
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

使用方法很简单。

Open可导入单张图片。

Open Dir可打开文件夹目录，然后可以用Next Image和Prev Image查看所有图片。

Change Save Dir可以更改xml文件保存的路径。

Verify Image可更改xml文件的内容。

Save可保存xml文件。

标注不要出现中文名字

### 开始训练自己模型

在该文件目录下面/home/demon/darknet/scripts，创建文件夹，VOCdevkit/,在该文件下面包含如下内容，

![undefined](http://ww1.sinaimg.cn/large/006lMPXUgy1gdtnob6hxtj30oq0fadh2.jpg)

JPEGImages放所有的训练图片，Aannotation放所有的xml标记文件。

ImageSets包含两个文件夹：layout和Main

在VOC2018下新建test.py文件夹，将下面代码拷贝进去运行，将在main文件下生成四个文件：train.txt,val.txt,test.txt和trainval.txt。

```Python
import os
import random

trainval_percent = 0.5
train_percent = 0.
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4]+ '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

### 开始编译

YOLOV3使用一个开源的神经网络框架Darknet53，使用C和CUDA，有CPU和GPU两种模式。默认使用的是CPU模式，需要切换GPU模型的话，vim修改Makefile文件。

**修改makefile文件，修改前5行内容和第50行，选择cuda的路径**



YOLOV3的label标注的一行五个数分别代表类别（从 0 开始编号）， BoundingBox 中心 X 坐标，中心 Y 坐标，宽，高。这些坐标都是 0～1 的相对坐标。label.py文件的目的是将xm文件转为txt文件。

**label.py源代码如下所示：**

```Python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2018', 'train'), ('2018', 'val'),('2018', 'test')]
classes = ["car"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().split("\n")


    list_file = open('%s_%s.txt'%(year, image_set), 'w')

    total = len(image_ids)
    index = 0
    for image_id in image_ids:
        index += 1
        if index == total - 1:
            break
        # print(image_id)
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

os.system("cat 2018_train.txt 2018_val.txt  > train.txt")
os.system("cat 2018_train.txt 2018_val.txt 2018_test.txt  > train.all.txt")

```

这里需要修改两个地方，sets和classes，classes根据自己需要修改。**修改了35和36行。**即` difficult = obj.find('difficult').text`，我添加了注释。

接下来运行该文件，我们的目录下会生成三个txt文件2018_train.txt,2018_val.txt,2018_test.txt，VOCdevkit下的VOC2018也会多生成一个labels文件夹，下面是真正会使用到的label，点开看发现已经转化成YOLOV3需要的格式了。这时候自己的数据集正式完成。

#### 修改cfg/voc.data

```yaml
classes= 1
train  = /home/demon/darknet/scripts/train.txt
valid  = /home/demon/darknet/scripts/2018_test.txt
names = data/voc.names
backup = backup
```

#### 修改data/voc.names和coco.names

car

#### 修改参数文件cfg/yolov3-voc.cfg

ctrl+f搜 yolo, 总共会搜出3个含有yolo的地方。

每个地方都必须要改2处， filters：3*（5+len（classes））；

其中：classes: len(classes) = 1，这里以单个类dog为例

filters = 18

classes = 1

可修改：random = 1：原来是1，显存小改为0。（是否要多尺度输出。）

参数文件开头的地方可以选训练的batchsize,

   修改20行的max_batches，表示迭代的次数

开始训练的指令

#### 开始训练的指令

`./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74`

**训练生成的文件保存在backup子目录下面**

测试结果指令

`./darknet detector test cfg/voc_RM.data cfg/yolov3-voc.cfg backup/yolov3-voc_900.weights data/1.png`

## 使用opencv驱动训练模型

```c++
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

string pro_dir = "/home/demon/CLionProjects/cv_test/"; //项目根目录

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 320;  // Width of network's input image
int inpHeight = 320; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression,使用非极大值抑制移除低置信度的边界框
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

void detect_image(string image_path, string modelWeights, string modelConfiguration, string classesFile);

void detect_video(string video_path, string modelWeights, string modelConfiguration, string classesFile);


int main(int argc, char** argv)
{
    // Give the configuration and weight files for the model
    String modelConfiguration = pro_dir + "yolov3-voc.cfg";
    String modelWeights = pro_dir + "yolov3-voc_final.weights";
    string image_path = pro_dir + "data/1.png";
    string classesFile = pro_dir + "voc.names";// "coco.names";
    //detect_image(image_path, modelWeights, modelConfiguration, classesFile);
    string video_path = pro_dir + "1223_90_radar.avi";

    detect_video(video_path, modelWeights, modelConfiguration, classesFile);
    cv::waitKey(0);
    return 0;
}

void detect_image(string image_path, string modelWeights, string modelConfiguration, string classesFile) {
    // Load names of classes
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    cv::Mat frame = cv::imread(image_path);
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Stop the program if reached end of video
    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    // Write the frame with the detection boxes
    imshow(kWinName, frame);
    cv::waitKey(30);
}

void detect_video(string video_path, string modelWeights, string modelConfiguration, string classesFile) {
    string outputFile  = "./yolo_out_cpp.avi";;
    // Load names of classes
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);    //存储包含种类信息的一行

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);  //DNN_TARGET_CPU

    // Open a video file or an image file 。or a camera stream.
    VideoCapture cap;
    //VideoWriter video;
    Mat frame, blob;

    try {
        // Open the video file
        ifstream ifile(video_path);
        if (!ifile) throw("error");
        cap.open(video_path);
    }
    catch (...) {
        cout << "Could not open the input image/video stream" << endl;
        return ;
    }

    // Get the video writer initialized to save the output video
    //video.open(outputFile,
    // VideoWriter::fourcc('M', 'J', 'P', 'G'),
    // 28,
    // Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        //video.write(detectedFrame);
        imshow(kWinName, frame);

    }

    cap.release();
    //video.release();

}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
```

