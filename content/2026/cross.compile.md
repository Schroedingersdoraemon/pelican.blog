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
