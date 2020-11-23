---
title: svm在opencv里面的使用
description: 
categories:
 - RM
tags:
---







SVM三宝：间隔、对偶、核技巧，SVM通过超平面对物体进行分类。

对SVM进行整理

## 参考代码

参考代码，实现功能是二分类

```c++
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/ml.hpp>
#include <string>
#include <iostream>

//1、描述
//打乱数据集
//2、输入
//matrix：打乱前数据集
//3、输出
//返回：打乱后数据集，也就是行与行互换
cv::Mat shuffleRows(const cv::Mat &matrix)
{
    std::vector <int> seeds;
    for (int cont = 0; cont < matrix.rows; cont++)        //将0~7存入到seeds里面
        seeds.push_back(cont);

//作用：将原数组（矩阵）打乱;
//参数：输入输出数组（一维），决定交换数值的行列的位置的一个系数，（可选）随机数产生器，0表示使用默认的随机数产生器，即seed=-1。rng决定了打乱的方法
    cv::randShuffle(seeds, 7);
    cv::Mat output;
    for (int cont = 0; cont < matrix.rows; cont++)  //打乱后的数据重新存入矩阵中
        output.push_back(matrix.row(seeds[cont]));

    return output;
}

//1、描述
//从文件夹中读取图像并标记，获得数据集hog特征
//2、输入
//type：文件夹路径
//3、输出
//des：数据集包含hog特征
//labels：数据集的标记
void readData(cv::Mat &des, cv::Mat &labels, int num, std::string type)
{
    for(int j = 0; j < 6; j++)
    {
        std::string path0 = type, temp0;
        std::stringstream ss1;              //类型转换
        ss1<<j;                             //向流中传值
        ss1>>temp0;                         //向temp0中写值
        path0.append(temp0);                //添加temp0
        path0.append("/");               //在temp0后面添加 /

        for (int i = 0; i < num; ++i)
        {
            //生成每张jpg图片的路径
            std::string path1 = path0, temp;
            std::stringstream ss2;
            ss2<<i;
            ss2>>temp;
            path1.append(temp);
            path1.append(".bmp");
            std::vector<float> des_temp;                      //存放结果,为HOG描述子向量
            std::cout<<path1<<std::endl;                       //输出读取图片的路径
            //图像预处理，方式必须和自动瞄准预处理相同
            cv::Mat img = cv::imread(path1, 1);        //读取图片
            cvtColor(img, img, cv::COLOR_BGR2GRAY);    //转为灰度图像
            medianBlur(img, img, 3);                 //中值滤波

            //HOG计算方法
            //检测窗口(40,40),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9  ，需要修改
            cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(40, 40), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
            //Hog特征计算，检测窗口移动步长(1,1)
            hog->compute(img, des_temp, cv::Size(1, 1), cv::Size(0, 0)); 
            
            //一行数据
            cv::Mat des_temp1(1, int(des_temp.size()), CV_32FC1, des_temp.data());     
            des.push_back(des_temp1);                                    //将数据压缩至des中

            //对数据集进行标记
            if (j == 0)                                                  
                labels.push_back(float(0));
            else
              labels.push_back(float(1));                                 
//            labels.push_back(float(j));
        }
    }
}

int main()
{
    //训练SVM分类器
    cv::Mat train_data;                  //数据
    cv::Mat train_labels;                //标签
    int num_0 = 3731;                    //数据总量
    int hog_num = 576;                  //hog长度
    //读取数据并且标记标签
    readData(train_data, train_labels, num_0, "/home/demon/CLionProjects/zhang_new_SVM/TrainData/");   //读取训练集图片，并提取HOG特征

    std::cout << train_data.size << std::endl;   //每一副图像形成一行,3731*6=22386,22386*576
    std::cout << train_labels.size << std::endl; //22386*1,一列
    cv::hconcat(train_data, train_labels, train_data);  //矩阵拼接，水平拼接
    train_data = shuffleRows(train_data);       //打乱数据集
    train_labels = train_data.rowRange(0, num_0*6).colRange(hog_num, hog_num+1);  //取特定行,包括最后，不包括前面

    train_labels.convertTo(train_labels, 4);  //矩阵数据类型转换
 
   train_data = train_data.rowRange(0, num_0*6).colRange(0, hog_num);
       //训练参数 关于SVM参数挺详细的解释:https://blog.csdn.net/computerme/article/details/38677599
       // 创建分类器并设置参数
            cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create(); // 创建分类器
            svm->setType(cv::ml::SVM::C_SVC);             //C_SVC用于分类，C_SVR用于回归
            svm->setKernel(cv::ml::SVM::KernelTypes::RBF); //选用的核函数
            svm->setGamma(0.01);//核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数;
            svm->setC(10.0);   //SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数；
           //设置终止条件，训练次数最多3000次，或者误差小于1e-6
            svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 3000, 0.1，1e-6));
           //  
           //该类(TermCriteria)变量需要3个参数，一个是类型，第二个参数为迭代的最大次数，最后一个是特定的阈值。
          //类型有COUNT, EPS or COUNT + EPS
          //分别代表着迭代终止条件为达到最大迭代次数终止，迭代到阈值终止，或者两者都作为迭代终止条件。
            std::cout << "参数设置完成" << std::endl;
            std::cout << "开始训练分类器" << std::endl;
           //训练数据和标签的结合
            cv::Ptr<cv::ml::TrainData>Inputdata = cv::ml::TrainData::create(train_data,cv::ml::ROW_SAMPLE,train_labels);
            svm->train( Inputdata);//layout:ROW_SAMPLE 为数据排列的方式，即行为一个样本
            std::cout << "分类器训练完成" << std::endl;

            //保存训练器
            svm->save("my_11_1_autohit.xml");
            std::cout << "my_11_1_autohit.xml" << std::endl;
            
/*********************************************以下属于测试部分******************************/

//载入分类器并测试
std::cout << "开始导入分类器文件...\n";
cv::Ptr<cv::ml::SVM> svm1 = cv::ml::StatModel::load<cv::ml::SVM>("my_11_1_autohit.xml");
std::cout << "成功导入分类器文件...\n";
// 测试分类器准确率
    int num =12;                                                                     //每类测试集有12张图片
    int num_img = 0;
    float count = 0;
    for(int j = 0; j < 6; j++)                                                       //测试的种类
    {
        std::string type = "/home/demon/CLionProjects/zhang_new_SVM/testimage/";       //测试集路径
        std::string path0 = type, temp0;                                        
        std::stringstream ss1;                                                       //数据类型转换
        ss1<<j;
        ss1>>temp0;
        path0.append(temp0);                                                 
        path0.append("/");
                                                                     
        for (int i = 0; i < num; ++i)
        {
            std::string temp1 = temp0;
            int result;
            if (j == 0)                                              //测试集实际结果
                result = 0;
            else
                result = 1;

            std::string path1 = path0, id;
            std::stringstream ss2;                               //每类测试集的图片名字是0~11.bmp
            ss2<<i;
            ss2>>id;
            temp1.append(id);
            path1.append(id);
            path1.append(".bmp");
            temp1.append(".bmp");
            std::vector<float> des_temp;                         //存储特征子向量
            std::cout<<path1<< std::endl;
            cv::Mat img = cv::imread(path1, 1);
            cvtColor(img, img, cv::COLOR_BGR2GRAY);              //灰度转换
            medianBlur(img, img, 3);
            //同上面解释
            cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(40, 40), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
            hog->compute(img, des_temp, cv::Size(1, 1), cv::Size(0, 0));
            std::cout<<i<<std::endl;

            float pre;
            //图片存储路径
            std::string path2 = "/home/demon/CLionProjects/zhang_new_SVM/True",path3 = "/home/demon/CLionProjects/zhang_new_SVM/Flase", labels, pres;
            ss2.clear();
            ss2<<result;
            ss2>>labels;
            path3.append(labels + "_" );
            path3.append(id + "_");
            pre = svm1->predict(des_temp);
            ss1.clear();
            ss1<<pre;
            ss1>>pres;

            path3.append(pres + ".jpg");

            std::cout<<"True labels: "<<float(result)<<"  "<<"Predicted labels: "<<pre<<std::endl;
            count += (pre == float(result));

            if (pre == float(result))
            {
//                imwrite(path2, img);
            } else
            {
                imwrite(path3, img);
            }
            num_img++;
        }
    }
    std::cout << "正确的识别个数： " << count <<  std::endl;
    std::cout << "正确率为：" << (count / num_img) * 100 << "%\n";
    std::cout<<num_img;
    return 0;
}
```

## 解释

（二分类）以上这段代码实现的是识别是杂乱无章的图片还是正常图片，也就是杂乱无章图片返回为0，1~5的数字返回是1，从而判断正确率。也可以进行多分类问题

本人测试集采用的是0文件下存储垃圾图片，1~5文件夹下存储的是1~5的数字，每类文件下存储3731张图片。测试集则是每类文件下存储12张图片。

整体流程是：1，对图片进行预处理（转为灰度图像、中值滤波等）

​                     2，提取HOG特征，得到HOG描述子向量

​                     3，对训练集进行遍历操作，完成以上两个步骤

​                     4，打乱数据集、矩阵拼接，重新得到训练数据和标签（存疑）

​                     5，创建分类器并设置SVM参数，包括:

 （1）设置SVM是用于分类还是回归 ---------------------------------------------------- svm->setType

 （2）选用的核函数，以上选用的是径向基函数------------------------------------svm->setKernel

 （3）核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数;----------------svm->setGamma

 （4）最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数---------------- svm->setC

 （5）设置终止条件，比如通过训练的次数和误差来判断---------------------------svm->setTermCriteria

​                      （6） 将训练数据和标签的结合 -----------------------------------------------------------cv::ml::TrainData::create    

​                      （7） //保存训练器----------------------------------------------------------------------------- svm->save

​                    6，测试分类器准确率

​                       （1）预处理图像，提取HOG特征，得到描述子向量

​                        （2） pre = svm1->predict(des_temp); 得到预测结果

​                         （3）通过与实际结果进行比对得到准确率

二、对于设置SVM参数部分也可以使用svm->trainAuto的方式进行训练，其优势是自动帮忙优化参数。

```c++
// 创建分类器并设置参数    
cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();    
svm->setType(cv::ml::SVM::C_SVC); //C_SVC用于分类，C_SVR用于回归    
svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR); //选用的核函数     
std::cout << "参数设置完成" << std::endl; //训练分类器    
std::cout << "开始训练分类器" << std::endl;    
//svm自动优化参数,同时将标签和数据进行结合    
svm->trainAuto(train_data, cv::ml::ROW_SAMPLE, train_labels);
std::cout << "分类器训练完成" << std::endl; //保存训练器    
svm->save("my_11_1_autohit.xml");   
std::cout << "my_11_1_autohit.xml" << std::endl; 
```



核函数包含以下类型：

![undefined](http://ww1.sinaimg.cn/large/006lMPXUgy1ge2pnzduwtj30ow0b5ab9.jpg)

## 注意：

SVM的训练函数是ROW_SAMPLE类型的，也就是说，送入SVM训练的特征需要reshape成一个行向量，所有训练数据全部保存在一个Mat中，一个训练样本就是Mat中的一行，最后还要讲这个Mat转换成CV_32F类型，例如，如果有𝑘个样本，每个样本原本维度是(ℎ,𝑤)，则转换后Mat的维度为(𝑘,ℎ∗𝑤)

对于多分类问题，label矩阵的行数要与样本数量一致，也就是每个样本要在label矩阵中有一个对应的标签，label的列数为1，因为对于一个样本，SVM输出一个值，我们在训练前需要做的就是设计这个值与样本的对应关系。对于有𝑘个样本的情况，label的维度是(𝑘,1)

​                                       