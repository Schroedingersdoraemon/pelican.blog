---
title: cross compile
date: 2026-04-16 16:54
tags:
---

[TOC]

# 0. what is

在 A 平台上编译某代码的结果，只能在 B 平台运行，此之谓交叉编译。

由于 arm 设备性能孱弱，因此交叉编译常发生于嵌入式环境。

由 amd64 的 HOST 为 arm64 的 TARGET 编译。

古代交叉编译常常涉及到 Gentoo 的 crossdev，例如 aarch64-unknown-linux-gnu-{gcc,g++}

在现代， LLVM 是更优先的选择。clang 具有 LLVM 支持平台的机器码生成能力。


> Target Selection Options
> 
>     Clang fully supports cross compilation as an inherent part of its design.
>     Depending on how your version of Clang is configured, it may have support
>     for a number of  cross  compilers, or may only support a native target.
>     
>     -arch <architecture>
>             Specify the architecture to build for (Mac OS X specific).
>     
>     -target <architecture>
>             Specify the architecture to build for (all platforms).


在任何平台上，你都可以使用 clang 生成 任意平台的代码。方法是传递 -target 参数。

比如

clang++ -target aarch64-unknow-linux-gnu main.cpp

不出任何意外的，意外发生了。

clang 能成功编译，但是在 最后 连接 的阶段报错了。会提示找不到 crtbeginS.o 之类的文件。

其实很好理解。 clang 虽然拥有生成任意平台的二进制的能力，但是，编译并不只是产生目标文件。更重要的是，还需要连接到运行时库。

那么，这个运行时库要怎么获取呢？ 答案是： 下载 alarm 的 根目录 包。

比如下载 ArchLinuxARM-aarch64-latest.tar.gz

将 ArchLinuxARM-aarch64-latest.tar.gz 解压到 /usr/gnemul/qemu-aarch64/

sudo tar -xvf ArchLinuxARM-aarch64-latest.tar.gz -C /usr/gnemul/qemu-aarch64/

然后使用这个命令编译

clang++ -target aarch64-unknow-linux-gnu --sysroot=/usr/gnemul/qemu-aarch64/ main.cpp

恭喜你，这次成功编译了。（截至今日， archlinuxarm 上带的 STL 是个有 bug 的版本，导致它的头文件有错误。见 bug, 如果遇到了，请相信我，不是我教的方法的问题，是真的系统带的头文件有 bug。自己按bug汇报修下吧。）

对于支持将 “clang++ -target aarch64-unknow-linux-gnu –sysroot=/usr/gnemul/qemu-aarch64/” 作为编译器的 autotools 工具来说，这个教程已经结束了。

因为只要配置环境变量

export CC="clang -target aarch64-unknow-linux-gnu --sysroot=/usr/gnemul/qemu-aarch64/"
export CXX="clang++ -target aarch64-unknow-linux-gnu --sysroot=/usr/gnemul/qemu-aarch64/"

autotools 系列工具就能正常运行了。

但是，同样的方法在 cmake 上会失效。因为 cmake 会把 “clang++ -target aarch64-unknow-linux-gnu –sysroot=/usr/gnemul/qemu-aarch64/” 作为一个整体去调用可执行文件。当然，系统里并不存在一个名为 “clang++ -target aarch64-unknow-linux-gnu –sysroot=/usr/gnemul/qemu-aarch64/” 的可执行文件。。。。

对 cmake 来说，还得多做一个工作，就是建立一个 wrapper。

比如写一个 aarch64-unknow-linux-gnu-clang 的脚本，脚本里面这么写

exec clang -target aarch64-unknow-linux-gnu --sysroot=/usr/gnemul/qemu-aarch64/ $*

还有写一个 aarch64-unknow-linux-gnu-clang++ 的脚本，脚本里面这么写

exec clang++ -target aarch64-unknow-linux-gnu --sysroot=/usr/gnemul/qemu-aarch64/ $*

这样，就可以如传统的交叉编译器做法一样，让 cmake 使用 aarch64-unknow-linux-gnu-clang 和 aarch64-unknow-linux-gnu-clang++ 作为编译器进行交叉。
