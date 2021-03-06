---
title: 搭建opencv的GPU环境
description: 想用opencv加载yolo模型，但是如果只用cpu的话速度会很慢，所以必须为电脑搭建环境
categories:
 - RM
 - opencv
 - 深度学习
tags:
---

在RM圆桌会议出来之前，对于雷达的具体定位一直不清晰，尝试使用过yolov3，为了提高运行速度进行安装CUDA和Cudnn。

## 背景

想使用opencv驱动yolov3训练的模型，但是只使用CPU的时候，发现一个问题，速度太慢，使用opencl的时候反而时间更长，貌似原因是需要交互所以需要时间更长（原谅我不求甚解),于是想要gpu进行加速，但这需要使用cuda,，但是发现使用opencv4.1.0不能使用cudnn，而opencv4.2.0则可以。所以重装opencv4.2.0。

系统：ubuntu16.04

电脑显卡：GTX850M



## 安装opencv4.2.0以及扩展包 

#### 安装包下载

第一步首先进行安装包下载，分别包括opencv和contribute，可以在[opencv4.2.0](https://docs.opencv.org/)上下载opencv源码和github上下载[contribute](https://github.com/opencv/opencv_contrib)，这里注意一点，opencv和扩展包要保持一致。比如你使用opencv4.2.0的源码，那么扩展包必须也是4.2.0版本。将opencv解压后放置在home下，同时将扩展包也放置在opencv目录下面，并在opencv目录下建立build文件。

#### 配置CMAKE

##### 使用cmake-gui配置cmake更简单

```    powershell
sudo apt-get install cmake-gui   #安装指令 
cmake-gui                        #打开方法  
```

##### 安装opencv相关依赖包

``` powershell
sudo apt-get install build-essential cmake git
sudo apt-get install libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev  
sudo apt-get install python-dev python-numpy  python3-dev python3-numpy   
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get -y install libavresample-dev
sudo apt-get -y install libgstreamer-plugins-base1.0-dev
sudo apt-get -y install libgstreamer1.0-dev
sudo apt-get install libopenblas-dev
sudo apt-get install doxygen
sudo apt install ccache
sudo apt install openjdk-8-jdk
sudo apt install pylint
sudo apt-get install liblapacke-dev checkinstall
```

```bash
#为了安装 Jasper
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install libjasper1 libjasper-dev
```



##### 操作cmake-gui界面

```
- 主要修改CMAKE_BUILD_TYPE为RELEASE

-勾选 OPENCV_GENERATE_PKGCONFIG

- 修改OPENCV_EXTRA_MODULES_PATH为
  /home/demon/RobMaster/opencv420/opencv_contrib-4.2.0/modules
  
-勾选OPENCV_GENERATE_PKGCONFIG和OPENCV_ENABLE_NONFREE.
  目的是生成pkg文件，以及可以使用一些需要授权的文件。
  
- 勾选WITH_CUDA和OPENCV_DNN_CUDA
```

![cmake-gui.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdrfy717phj30gq0hmac2.jpg)

##### 修改文件，避免翻墙

```
- gedit /home/demon/robmasteer/opencv420/3rdparty/ippicv/ippicv.cmake
记得demon换成自己的用户名， 将47行改为文件的本地路径："file:///home/demon/下载/"（仅供参考，根据自己的路径填写，下载的是加速文件）

- 修改opencv420/opencv_contrib-4.2.0/modules/face/CMakeLists.txt第19行，下载face_landmark_model.dat文件

- 修改opencv420/opencv_contrib-4.2.0/modules/xfeatures2d/cmake/download_boostdesc.cmake的27行，下载boostdec_bgm等文件

- 修改opencv420/opencv_contrib-4.2.0/modules/xfeatures2d/cmake/download_vgg.cmake21行，下载vgg_generated_120.i 等文件
```

##### 修改CMAKE匹配的GPU计算能力

  opencv4.2.0对GPU计算能力要求5.3以上，所以删除不匹配的计算能力，如图所示。

![computer.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdrg9dlil2j30ga0lsmzv.jpg)

##### 点击configure和generate，完成cmake配置

#### 开始编译

```powershell
sudo make -j8   #8核编译 
sudo make install #安装
```
在这里，我的朋友说我没有特别说明这个。导致他只用了`sudo make `,然后编译时间较长，所以，我要在此声明---------------------后面一定要加参数`-j8`

配置一些OpenCV的编译环境,  将OpenCV的库添加到路径，从而可以让系统找到

 `sudo gedit /etc/ld.so.conf.d/opencv.conf  `

  新建或者打开此文件，在文件末尾添加`/usr/local/lib`

##### 配置bash

`sudo gedit /etc/bash.bashrc    `

在该文件下面添加如下内容：

```powershell
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
export PKG_CONFIG_PATH   
```
##### 配置生效

```powershell
source /etc/bash.bashrc    #配置生效
sudo updatedb              #更新       
```

## 安装CUDA

#### 查看电脑配置

```powershell
lspci |grep VGA
ubuntu-drivers devices
```

![截图.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdrgyn9bdij30jx0bhwfx.jpg)

可以看到我的电脑的显卡是GTX 850M,推荐安装的驱动是 nvidia-440

#### 安装显卡驱动法一

如果同意安装推荐驱动，则可以输入如下指令：

```sudo ubuntu-drivers autoinstal```

也可以选择自己需要的

```sudo apt install nvidia-384```

#### 安装显卡驱动法二

##### 下载驱动

进入[NVDIA](https://www.nvidia.com/Download/index.aspx)下载驱动，根据我的个人电脑进行如下选择

![NVDIA.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdrh6asldtj30kc09sgmd.jpg)

##### 处理原有的驱动

`sudo apt-get remove --purge nvidia*`

禁用Ubuntu默认自带安装的开源驱动：nouveau

`sudo gedit /etc/modprobe.d/blacklist.conf`

在次文件中添加以下内容：

```powershell
blacklist nouveau 
blacklist lbm-nouveau 
options nouveau modeset=0 
alias nouveau off alias 
lbm-nouveau off
```

```powershell
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf#关闭nouveau
update-initramfs -u
reboot  #重启电脑
```
##### 验证是否成功

```lsmod | grep nouveau```

#### 安装CUDA驱动

ctrl+alt+f1进入字符界面，ctrl+alt+f7返回界面

```powershell
sudo service lightdm stop      #这个是关闭图形界面，不执行会出错
sudo chmod  a+x NVIDIA-Linux-x86_64-396.18.run  # 给驱动run文件赋予执行权限
sudo ./NVIDIA-Linux-x86_64-396.18.run -no-x-check -no-nouveau-check -no-opengl-files #安装
```

只有禁用opengl这样安装才不会出现循环登陆的问题

-no-x-check：安装驱动时关闭X服务

-no-nouveau-check：安装驱动时禁用nouveau

-no-opengl-files：只安装驱动文件，不安装OpenGL文件

##### 安装过程

- The distribution-provided pre-install script failed! Are you sure you want to continue? **选择 yes 继续。**

- Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later? **选择 No 继续。**

- 问题没记住，选项是：**install without signing**

- 问题大概是：Nvidia’s 32-bit compatibility libraries? **选择 No 继续。**

  Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up. **选择 Yes 继续**
  
##### 打开图形界面

```sudo service lightdm start      #这个是打开图形界面```

#### 安装CUDA

##### 下载源文件

进入官网下载[cudnn](https://developer.nvidia.com/cuda-release-candidate-download),本人下载的是cuda_10.2.89_440.33.01_linux.run和cudnn-10.2-linux-x64-v7.6.5.32.tgz.

cuda和cudnn,二者一定要对应.

##### 安装cuda

```powershell
sudo chmod +x cuda_10.2.89_440.33.01_linux.run  #改变可读可写权限
sudo ./cuda_10.2.89_440.33.01_linux.run       
```

驱动已经安装就不要在安装了，剩下的按照默认的就行了.

![截图.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdri1r4m1kj30k30btq3s.jpg)

##### 配置环境

```sudo gedit ~/.bashrc```

将安装路径添加到文件末尾：

```powershell
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

然后执行如下命令使路径生效：

```~/.bashrc```

##### 验证是否成功

```nvcc -V```

出现下面图形说明安装成功

![computer.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdri681rxwj30k20cq3z0.jpg)

使用自带的例子进行测试：

```powershell
cd /usr/local/cuda-10.2/samples/1_Utilities/deviceQuery
sudo make        #编译代码
./deviceQuery   #执行
```

出现如下结果说明安装成功

![截图.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdri7v3e3yj30jm0cggng.jpg)

## 安装CUdnn

##### 过程

```powershell
tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
```

##### 验证

```nvidia-smi ```

![computer.png](http://ww1.sinaimg.cn/large/006lMPXUgy1gdria6erp6j30k708lmxv.jpg)

## 问题补充

1，在使用yolo训练出来的模型（迭代900次 ），发现使用有问题，主要问题如下，使用cpu处理320*320图片需要600ms作用，使用opencl需要2s左右，使用cuda需要不到100ms处理一帧图片。

2，在使用cmake-gui生成cuda文件时候，提示一个错误，大概就是opencv使用cuda,要求nvidia显卡计算能力达到5.3以上，显卡的[计算能力对应表格](https://developer.nvidia.com/cuda-gpus#compute)







**Demon 说**： 我同门在安装opencv过程中遇到的问题补充，cmake-gui界面出现 的报错信息，比如找不到什么什么文件时候，一定要把缺失文件安装上，否则在make界面还会有问题的。







参考[链接](https://www.jianshu.com/p/259a6140da9d)