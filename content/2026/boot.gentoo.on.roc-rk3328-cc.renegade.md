---
title: boot gentoo on roc-rk3328-cc renegade
date: 2026-05-03 11:25
tags: arm, gentoo, cross compile
---

[TOC]

# 0. boot

boot 有两种方法

- [从头编译uboot](#00-compile-uboot)
- [拿来主义：巧得二进制](#01-optional-retrieve-armbian-boot)

## 0.0. compile uboot

```shell
git clone --depth=1 https://github.com/u-boot/u-boot
git clone --depth 1 https://github.com/rockchip-linux/rkbin.git
```

把 rk322xh_bl31_v1.49.elf 和 rk3328_ddr_400MHz_V1.21.bin 拿出来

```shell
# 这里放到了 u-boot 里面
mkdir u-boot/external.rockchip
cp rkbin/bin/rk33/{rk322xh_bl31_v1.49.elf,rk3328_ddr_400MHz_V1.21.bin} \
    u-boot/external.rockchip
```

这两个环境变量很重要，否则无法编译成功

```shell
cd u-boot/
export BL31=./external.rockchip/rk322xh_bl31_v1.49.elf
export ROCKCHIP_TPL=./external.rockchip/rk3328_ddr_400MHz_v1.21.bin
```

根据文件 configs/roc-cc-rk3328_defconfig

```shell
make roc-cc-rk3328_defconfig
```

u-boot 现在需要 as 等工具，所以只能暂且使用 crossdev

```shell
doas crossdev --stage2 --target aarch64-unknown-linux-gnu
```

如此这般就获得了 aarch64-unknown-linux-gnu- 系列的工具

编译之前请检查 pyelftools，gentoo 可以通过 app-misc/paxutils python 避免增加 world 条目
否则不会产生 u-boot.itb，且会报错如下：

> Wrote map file './simple-bin.map' to show errors
> binman: Node '/binman/simple-bin/fit': subnode 'images/@atf-SEQ':
> Failed to read ELF file: Python: No module named 'elftools'

最后编译如下：

```shell
make HOSTCC=clang LLVM=1 CROSS_COMPILE=aarch64-unknown-linux-gnu- CC=clang -j10
```

获得了 **idbloader.img** 和 **u-boot.itb**。

再根据 rockchip 官网处的 [Default storage map](https://opensource.rock-chips.com/wiki_Partitions)

```shell
# 烧录 idbloader.img 到偏移量 32KB 处(对应扇区64)
doas dd if=idbloader.img of=/dev/sdX conv=fsync,notrunc bs=512 seek=64
# 烧录 u-boot.itb 到偏移量 8MB 处(对应扇区16384)
doas dd if=u-boot.itb of=/dev/sdX conv=fsync,notrunc bs=512 seek=16384
```

| Partition      | start sec | Start(h) | Number of Sectors | Secs(h)  | Part Size | Size   | PartNum in GPT | Requirements                         |
| -------------- | --------- | -------- | ----------------- | -------- | --------- | ------ | -------------- | ------------------------------------ |
| MBR            | 0         | 00000000 | 1                 | 00000001 | 512       | 0.5KB  |                |                                      |
| Primary GPT    | 1         | 00000001 | 63                | 0000003F | 32256     | 31.5KB |                |                                      |
| loader1        | 64        | 00000040 | 7104              | 00001bc0 | 4096000   | 2.5MB  | 1              | preloader (miniloader or U-Boot SPL) |
| Vendor Storage | 7168      | 00001c00 | 512               | 00000200 | 262144    | 256KB  |                | SN, MAC and etc.                     |
| Reserved Space | 7680      | 00001e00 | 384               | 00000180 | 196608    | 192KB  |                | Not used                             |
| reserved1      | 8064      | 00001f80 | 128               | 00000080 | 65536     | 64KB   |                | legacy DRM key                       |
| U-Boot ENV     | 8128      | 00001fc0 | 64                | 00000040 | 32768     | 32KB   |                |                                      |
| reserved2      | 8192      | 00002000 | 8192              | 00002000 | 4194304   | 4MB    |                | legacy parameter                     |
| loader2        | 16384     | 00004000 | 8192              | 00002000 | 4194304   | 4MB    | 2              | U-Boot or UEFI                       |
| trust          | 24576     | 00006000 | 8192              | 00002000 | 4194304   | 4MB    | 3              | trusted-os like ATF, OP-TEE          |
| boot（must）   | 32768     | 00008000 | 229376            | 00038000 | 117440512 | 112MB  | 4              | kernel, dtb, extlinux.conf, ramdisk  |
| rootfs         | 262144    | 00040000 | -                 | -        | -         | -MB    | 5              | Linux system                         |
| Secondary GPT  | 16777183  | 00FFFFDF | 33                | 00000021 | 16896     | 16.5KB |                |                                      |

## 0.1. (optional) retrieve armbian boot

首先获取 [armbian renegade noble](https://mirrors.bfsu.edu.cn/armbian-releases/renegade/archive)

> Noble Ubuntu 24.04 LTS  
> Resolute Ubuntu 26.04  
> Trixie Debian 13  
> _help yourself_

可以拿出固件部分，也可以直接 pipe 到 of=/dev/sdX

```shell
xz -d -c ./Armbian_26.2.1_Renegade_noble_current_6.18.8_minimal.img.xz \
    | doas dd of=rk3328_boot.bin bs=512 count=32768
```

# 1. flash your card

初始化 tf 卡，数据无价，谨慎操作！

```shell
doas fdisk /dev/sdb
g # new GPT disklabel
w # write
q # quit
```

为 /dev/sdX 设备 dd 写入 固件分区

```shell
doas dd if=./rk3328_boot.bin of=/dev/sdb bs=512 count=32768 status=progress
```

接着重建 gpt 分区表

**不要 remove 现存 ext4 signature**

```shell
doas fdisk /dev/sdb
g # new GPT disklabel
n # 创建新分区
1 # 分区号默认是1
32768 #        重要！ 从 16M （512kb/扇区 x 32768 扇区）处开始
回车 # 扩展到最后
n #           重要！ 不要擦除 signature
w # write
q # quit
```

最后 mkfs

```shell
doas mkfs.ext4 /dev/sdb1
# 如果寻求更高性能，且无需 log，可 doas mkfs.ext4 -O ^has_journal /dev/sdb1
```

> 请忽略此处  
> 如果直接 dd 某 img，那么此处可能需要这些指令。
>
> doas e2fsck -f /dev/sdb1  
> doas resize2fs /dev/sdb1

# 2. kernel

类似的，kernel 也有两种方法

- [编译 Image dtbs from scratch](#21-compile-image-and-dtbs)
- [拿来主义：获得现成发行产物](#22-optional-use-binary)

## 2.1. compile Image and dtbs

copy gentoo-sources to your workspace

```shell
make ARCH=arm64 defconfig
make ARCH=arm64 menuconfig
```

然后 platforms 只保留 rockchip platforms

再按照需求 disable 某些特性，例如 virtualization 和无用的驱动

```shell
make ARCH=arm64 LLVM=1 LLVM_IAS=0 \
     CROSS_COMPILE=aarch64-unknown-linux-gnu- \
     -j$(nproc) \
     Image dtbs modules
```

然后 arch/arm64/boot 处获得 Image 和 .../dts/rockchip/rk3328-roc-cc.dtb

再安装 modules

```shell
doas make ARCH=arm64 INSTALL_MOD_PATH=/mnt modules_install
```

如果需要 initramfs，除了 menuconfig 中麻烦配置。

还可以通过 systemd-nspawn 异构 chroot 中 mkinitcpio

> ls /lib/modules  
> 6.19.11-1-aarch64-ARCH 7.0.1-gentoo

```shell
mkinitcpio -k 7.0.1-gentoo -g initramfs-gentoo.img
```

然后把产物放在 /boot 中并修改 extlinux.conf

## 2.2. (optional) use binary

根据你的发行版偏好要求

- [gentoo](#221-gentoo)
- [archlinux](#222-alarm)

### 2.2.1. gentoo

#### 2.2.2.1. 获取 alarm 内核产物

选择镜像 [bfsu mirror](https://mirrors.bfsu.edu.cn/archlinuxarm/os/)
获得 ArchLinuxArm-aarch64-latest.tar.gz ， 并解压出 boot

```shell
tar xpfv ArchLinuxArm-aarch64-latest.tar.gz ./boot
# 复制 boot 的 Image initramfs-linux.img dtbs/rockchip/rk3328-roc-cc.dtb
cp ./boot/{Image,initramfs-linux.img,dtbs/rockchip/rk3328-roc-cc.dtb} /mnt
```

另外要把 arch rootfs 里面的 modules 拷贝过去

```shell
tar xpfv ArchLinuxArm-aarch64-latest.tar.gz ./usr/lib/modules
```

#### 2.2.2.2. gentoo stage3

grab gentoo stage3 files

选择镜像 [bfsu mirror](https://mirrors.bfsu.edu.cn/gentoo/releases)
获得 gentoo arm64 的 current-stage3-arm64-systemd 的 tar.xz 文件。

此处记得挂载分区到 /mnt，以 gentoo 之方式解压 stage3

```shell
doas tar xfpv stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner -C /mnt
```

### 2.2.2. alarm

当然，如果想避免麻烦，直接 archlinux arm 最为简洁。

```shell
doas bsdtar -xpf ./ArchLinuxARM-aarch64-latest.tar.gz -C /mnt
```

# 3. extlinux

```shell
cd /mnt/boot
doas mkdir extlinux
cd extlinux
doas tee extlinux.conf << EOF
label gentoo
    kernel /boot/Image
    initrd /boot/initramfs-linux.img
    fdt /boot/dtbs/rockchip/rk3328-roc-cc.dtb
    append root=PARTUUID=$(blkid /dev/sdb1) rw rootwait console=ttyS2,1500000

    # 除了 /boot 下对应的文件，切记 PARTUUID 和 UUID 的辨析。
EOF
```

也可这么写 append，但是记得删除换行，

> append earlycon=uart8250,mmio32,0xff130000 \
>  console=ttyS2,1500000 \
>  console=tty0root=PARTUUID=b921b045-1d \
>  rw rootwait rootfstype=ext4 loglevel=7

# 4. start

BOOM! 成功启动
