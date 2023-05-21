记录一次 Gentoo 的修缮过程
##########################
:date: 2023-04-29 13:29

.. contents::

0. abstract
===========

最近在 /etc/fstab 里面指定 /var/lib/docker 时出现了失误，致使我的 root 分区被污染，重新修复 seems implausible，于是痛定思痛重装 Gentoo，同时记录一下。


1. livecd
=========

1.1. mount
----------

即使 Gentoo 多年，由于我总是在 grub 保有 recovery entry，所以进 livecd 抢救的次数也屈指可数，这使我总是忘记这段关键的命令

.. code-block:: shell

   sudo mkdir /mnt/gentoo
   sudo mount -vo compress-force=zstd,subvolid=256 /dev/nvme0n1p3 /mnt/gentoo
   sudo mount -vo compress-force=zstd,subvolid=257 /dev/nvme0n1p3 /mnt/gentoo/home
   sudo mount /dev/nvme0n1p1 /mnt/gentoo/boot
   sudo swapon /dev/nvme0n1p2
   sudo mount --types proc /proc /mnt/gentoo/proc
   for i in sys dev run; do
   for > sudo mount --rbind /$i /mnt/gentoo/$i
   for > sudo mount --make-rslave /mnt/gentoo/$i
   done

1.2. stage3
-----------

下载 stage3 之前最重要的事，which I often forget，就是校准时间

.. code-block:: shell

   sudo date 042914362023 # 将时间设定为 04月29日，14：36，2023年

另一个就是保留权限解压 tar 包

.. code-block:: shell

   tar xfpv stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner

1.3. base
---------

拷贝 dns 信息

.. code-block:: shell

   cp --dereference /etc/resolv.conf /mnt/gentoo/etc/

以及gentoo 仓库的信息
   
.. code-block:: shell

   mkdir /mnt/gentoo/etc/portage/repos.conf
   cp /mnt/gentoo/usr/share/portage/config/repos.conf /mnt/gentoo/etc/portage/repos.conf/gentoo.conf

或许在 /etc/portage/repos.conf/gentoo.conf 里面可以加上 **sync-openpgp-key-refresh = no**
   

2. Btrfs for safety
===================

前面其实都是些废话，最重要的就是好好利用 **Copy on Write, CoW** 和 **snapshot** 特性，empowered by Btrfs

2.1. 清理碎片
-------------

谨记，Btrfs 命令支持 *最短去重前缀*

.. code-block:: shell

   # btrfs filesystem defragment -v /
   # btrfs fi de -v /  # 以上命令可以写成这样

2.2. 子卷快照
-------------

Btrfs 的子卷并不是块设备，而是独立可挂载的 POSIX filetree

.. code-block:: shell

   # btrfs subvolume snapshot <subvolume> { <subdir>/<name> | <subdir> }
   # btrfs sub snap -r ... # -r 意味着该快照只读

首先在 / 下创建专用于备份的文件夹，我称其为 *snapshoot* 哈哈哈

.. code-block:: shell

   # btrfs subvolume create /snapshoot

我将 home 文件夹设为了子卷，所以需要快照两次，因为 btrfs 的快照 **不是递归的** ，
子卷存在的地方会被映射为同名的空文件夹
   
.. code-block:: shell

   # btrfs subvolume snapshot -r /     /snapshoot/$(date +"%Y-%m-%d")-readonly-root
   # btrfs subvolume snapshot -r /home /snapshoot/$(date +"%Y-%m-%d")-readonly-home
