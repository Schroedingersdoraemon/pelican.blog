---
layout: blog
title: deploy darknet with CUDA and OpenCV support on Arch Linux
date: 2020-09-03 20:55:10
tags:
---

![predicted image](/files/deploy-darknet-with-CUDA-and-OpenCV-support/predictions.jpg)

# 0. abstract

Despite the fact that using [darknet](https://pjreddie.com/darknet/) is rather simple, there are still some points to focus on.

# 1. install the base

```shell
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```

It is possibile to run several jobs simultaenously if your system is multi-core/multi-processor. The easiest way to check the number of cores is `nproc --all` or `cat /proc/cpuinfo`, `lscpu` is also feasible.

```shell
make -j$(nproc)
```

Once compiled, the binary file is executed as below:

```shell
./darknet
```

# 2. compile with CUDA

Change the first line of *Makefile* into `GPU=1`, then change the content in brackets <...> listed below:

```
ifeq ($(GPU), 1) 
COMMON+= -DGPU -I <...>
CFLAGS+= -DGPU
LDFLAGS+= -L<...> -lcuda -lcudart -lcublas -lcurand
endif
```

For instance, the default instll folder of CUDA on Arch Linux is `/opt/cuda`

```
ifeq ($(GPU), 1) 
COMMON+= -DGPU -I /opt/cuda/include
CFLAGS+= -DGPU
LDFLAGS+= -L/opt/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif
```

Then we need to look up the [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability) of our GPU [here](https://developer.nvidia.com/en/cuda-gpus).

For instance, my GPU, *NVIDIA GeForece MX130*, is not listed in the form. ~~According to some articles, however, this card is exactly the same as *GeForce 940MX*, the compute capability of which, in that form, is 5.0.~~ According to [wikipedia](https://en.wikipedia.org/wiki/CUDA), the compute capability is exactly 6.1

Therefore, we need to change the architecture in *Makefile* for [cuda compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#toolkit-versioning).

In detail, change both the code to ten times of your compute capability, which is 61 for me.

```
# before
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
      #-gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# after
ARCH= -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
      #-gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

```

After that we can compile it again using `make -j$(nproc)`, and render darket with CUDA support.

[Yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) downloaded, you can detect object in a rather good speed as followed:

```shell
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

# 3. OpenCV support

By default darknet does not support OpenCV, which means you have to manually open the predicted picture generated in the folder.

Just open *Makefile* and enable it:

```
OPENCV=1
```

Notice that the dependency name of package`opencv` in Arch Linux is **opencv4**. You can have a test by `pkg-config --libs opencv4`.

```
# before
ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv`
endif

# after
ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv4` -lstdc++
COMMON+= `pkg-config --cflags opencv4`
endif
```

When we've happily modified the pkg-config and expected it to compile successfully, things happened: life is hard.

The compilation went error with

```
./src/image_opencv.cpp:12:1: error: ‘IplImage’ does not name a type
   12 | IplImage *image_to_ipl(image im)
      | ^~~~~~~~
compilation terminated due to -Wfatal-errors.
make: *** [Makefile:86: obj/image_opencv.o] Error 1
make: *** Waiting for unfinished jobs....
```

Some C APIs, which have been deprecated for a long time, were removed from the lateset OpenCV release. The solution is to remove references to the C APIs and use C++ API or to re-include the old C APIs from the OpenCV legacies.

Open src/image_opencv.cpp

```cpp
#include "opencv2/imgproc/imgproc_c.h"
```

and change `IplImage ipl = m` in function `mat_to_image(Mat m)` to
```cpp
IplImage ipl = cvIplImage(m)
```

also, in the function `*void_video_stream(...)`

```cpp
// before
if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
if(fps) cap->set(CV_CAP_PROP_FPS, w);

// after
if(w) cap->set(CAP_PROP_FRAME_WIDTH, w);
if(h) cap->set(CAP_PROP_FRAME_HEIGHT, w);
if(fps) cap->set(CAP_PROP_FPS, w);
```

and in the function `make_window(...)`
```cpp
// before
setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

// after
setWindowProperty(name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
```

Tadah! Finally we successfully compiled darknet with CUDA and OpenCV support in Arch Linux !

Now we can test our darknet using either

```shell
./darket imtest data/eagle.jpg

# pop up an instance image
```

or

```shell
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

# object detection of an instance image in yolov3 with [pretrained weights](https://pjreddie.com/media/files/yolov3.weights) in a rather good speed
```

# 4. possible errors

## CUDA Error: out of memory

When the terminal prompts

```
28 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
29 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
30 res   27                  76 x  76 x 256   ->    76 x  76 x 256
31 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
32 CUDA Error: out of memory
darknet: ./src/cuda.c:36: check_error: Assertion `0' failed.
[1]  635531 abort (core dumped)  ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

indicates that you may have run out of your GPU memory.

The solution is quite easy: decrease the number of images per batch in corresponding configuration file e.g `cfg/yolov3.cfg`

```
# before
batch = 64

# after
batch = 16
```

# 5. prediction result

```shell
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

```shell
# output:
data/dog.jpg: Predicted in 0.719306 seconds.
dog: 100%
truck: 92%
bicycle: 99%
```

![raw image](/files/deploy-darknet-with-CUDA-and-OpenCV-support/dog.jpg)

![predicted image](/files/deploy-darknet-with-CUDA-and-OpenCV-support/predictions.jpg)
