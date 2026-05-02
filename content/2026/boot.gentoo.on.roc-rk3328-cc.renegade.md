---
title: boot gentoo on roc-rk3328-cc renegade
date: 2026-04-18 20:55
tags:
---

[TOC]

# 0. pre

## 0.0. armbian

首先获取 [armbian renegade noble](https://mirrors.bfsu.edu.cn/armbian-releases/renegade/archive)

### 0.0.0. 获取 boot 分区

```shell
xz -d -c ./Armbian_26.2.1_Renegade_noble_current_6.18.8_minimal.img.xz | doas dd of=rk3328_boot.bin bs=512 count=32768
```

### 0.0.1. (可选) 体验 armbian

为 /dev/sdX 设备 dd 写入 armbian.img

```shell
xz -d -c ./Armbian_26.2.1_Renegade_noble_current_6.18.8_minimal.img.xz | doas dd of=/dev/sdb bs=4M progress
```

然后从 armbian 获取 boot 固件分区

```shell
doas dd if=/dev/sdb of=rk3328_boot.pre32768.bin bs=512 count=32768
```

然后为设备扩容，通过 fdisk 记录/dev/sdb1 扇区起始位置 32768。

```
doas fdisk /dev/sdb
```

然后删除 /dev/sdb1 ， **n** 增加分区，起始扇区 32768，默认最后扇区结束。

**不要 remove 现存 ext4 signature**

随即应用扩容

```shell
doas e2fsck -f /dev/sdb1
doas resize2fs /dev/sdb1
```

扩容后自动分区 /dev/sdb1 充当 root ， 并格式化。

## 0.1. 已有 boot 备份

```shell
doas dd if=./rk3328_boot.pre32768.bin of=/dev/sdb bs=1M conv=notrunc
```

并格式化

```shell
doas mkfs.ext4 /dev/sdb1
# 如果寻求更高性能可以 doas mkfs.ext4 -O ^has_journal /dev/sdb1
```

# 1. grab files

## 1.1. get gentoo stage3

选择镜像 [bfsu mirror](https://mirrors.bfsu.edu.cn/gentoo/releases)
获得 gentoo arm64 的 current-stage3-arm64-systemd 的 tar.xz 文件。

此处记得挂载分区到 /mnt，以 gentoo 之方式解压 stage3

```shell
doas tar xfpv stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner -C /mnt
```

## 1.2. get kernel initramfs and dtb

选择镜像 [bfsu mirror](https://mirrors.bfsu.edu.cn/archlinuxarm/os/) 
获得 ArchLinuxArm-aarch64-latest.tar.gz ， 并解压出 boot

```shell
tar xpfv ArchLinuxArm-aarch64-latest.tar.gz ./boot
```

复制 boot 的 Image initramfs-linux.img dtbs/rockchip/rk3328-roc-cc.dtb 到 /mnt/boot


另外要把 arch rootfs 里面的 modules 拷贝过去

```shell
tar xpfv ArchLinuxArm-aarch64-latest.tar.gz ./usr/lib/modules
```

### 1.3.3. （可选）plain archlinux arm

如果不想要 gentoo 的 stage3，也可直接 boot archlinux

```shell
doas bsdtar -xpf ./ArchLinuxARM-aarch64-latest.tar.gz -C /mnt
```

最后莫忘忘记 arch-chroot /mnt 为 root 添加密码

and ...

# 2. extlinux


```shell
cd /mnt/boot
doas mkdir extlinux
cd extlinux
doas tee extlinux.conf << EOF
LABEL gentoo
    LINUX /boot/Image
    INITRD /boot/initramfs-linux.img
    FDT /boot/rk3328-roc-cc.dtb  # 若没有移动文件，默认位置是 /boot/dtbs/rockchip/...
    APPEND earlycon=uart8250,mmio32,0xff130000 console=ttyS2,1500000 \
rw rootwait rootfstype=ext4 root=UUID=$(lsblk -f | grep sdb1 | awk '{print $4}')  # 请单独运行指令补充 UUID
EOF
```

BOOM! 成功启动

