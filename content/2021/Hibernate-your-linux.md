---
layout: blog
title: Hibernate your linux
date: 2021-08-13 14:49:42
tags:
---

# 0. abstract

- **Suspend to RAM**, usually called **suspend**, cuts power to most parts of the machine aside from the RAM, which is required to restore the machine's state. Because of the large power savings, it is advisable for laptops to automatically enter this mode when the computer is running on batteries and the lid is closed (or the user is inactive for some time).

- **Suspend to disk**, usually called **hibernate**, saves the machine's state into swap space and **completely powers off** the machine. When the machine is powered on, the state is restored. Until then, there is zero power consumption.

- **Suspend to both**, usually called **hybrid suspend**, saves the machine's state into swap space, but does not power off the machine. Instead, it invokes usual suspend to RAM. Therefore, if the battery is not depleted, the system can resume from RAM. If the battery is depleted, the system can be resumed from disk, which is much slower than resuming from RAM, but the machine's state has not been lost.


# 1. Low level interfaces

## 1.1 kernel (swsusp)

The most straightforward approach is to directly inform the in-kernel software suspend code (swsusp) to enter a suspended state; the exact method and state depends on the level of hardware support. On modern kernels, writing appropriate strings to /sys/power/state is the primary mechanism to trigger this suspend.

## 1.2 uswsusp

Userspace Software Suspend


# 2. High level interfaces

## 2.1 systemd

systemd provides native commands  for suspend, hibernate, hybrid suspend

# 3. Kernel Documentation

The kernel documentation of Power Management Interface for System Sleep is [here](https://www.kernel.org/doc/Documentation/power/interface.txt)

## 3.1 /sys/power/state

`/sys/power/state` is the system sleep state control file. In it listed a list of supported sleep states, encoded as:

- **freeze**, suspend to idle, is always supported.
- **disk**, suspend to disk, always supported too as long the kernel has been configured to support hibernation at all.
- **mem**, suspend to RAM, depends on the capability of the platform.
- **standby**, power-on suspend, depends on the capability of the platform.

## 3.2 /sys/power/disk

`sys/power/disk`  controls the operating mode of hibernation, or suspend to disk. It tells the kernel what to do next after the creation of hibernation image.  Currently selected option is printed in square brackets.

- **platform**, put the system into sleep using a platform-provided method, is only available if the platform provides a special mechanism to sleep the system after the creation of hibernation image.
- **shudown**, shut the system down.
- **reboot**, reboot the system
- **suspend**, trigger a Suspend-to-RAM transition, is only available if Suspend-to-RAM is supported
- **test_resume**, resume after hibernation test mode, is well illustrated in [Documentation/power/basic-pm-debugging.txt](https://www.kernel.org/doc/Documentation/power/basic-pm-debugging.txt)

## 3.3 /sys/power/image_size

`/sys/power/image_size` controls the size of hibernation images.

It can be written a string representing a non-negative integer that will be used as a best-effort upper limit of the image size, in bytes.  The hibernation
core will do its best to ensure that the image size will not exceed that number. However, if that turns out to be impossible to achieve, a hibernation image will still be created and its size will be as small as possible.  In particular, writing '0' to this file will enforce hibernation images to be as small as possible.

Reading from this file returns the current image size limit, which is set to around 2/5 of available RAM by default.

## 3.4 more

`/sys/power/pm_trace` and more information is [here](https://www.kernel.org/doc/Documentation/power/))

# 4. configuration

A swap partition or swap file is need. Then you will point the kernel to the *swap* using the `resume=` kernel parameter in oot loader. To configure the initramfs is also needed to tell the kernel.

## 4.1 swap size

According to [previous content](#kernel-documentation) of system sleep, a small swap partation is also very likely to hibernate successfully. By the way, you are *strongly* recommended to read the kernel documentation mentioned previously, which offers numerous help in a straightforward way.

The size of *swap_file* or *swap_partition* is recommended to be about **one to twice the RAM size**.

## 4.2 swap_file

According to [kernel documentation](https://www.kernel.org/doc/Documentation/power/swsusp-and-swap-files.txt), to use a *swap_file*, you need to:

1. Create the swap file and make it active,

   ```bash
   dd if=/dev/zero of=<path_to_swap_file> bs=1024 count=<swap_size_in_k>
   mkswap <path_to_swap_file>
   swapon <path_to_swap_file>
   ```

2. Use an application that will bmap the *swap_file* with the help of the FIBMAP ioctl and determine the location of the file's swap header, as the **offset**, in *PAGE_SIZE* units, from the beginning of the partition which holds the swap file.

3. Update kernel parameter

   ```
   resume=<swap_file_partition> resume_offset=<swap_file_offset>
   ```

   where *swap_file_partition* is the partition on which the swap file is located, and *swap_file_offset* is the offset of the swap header determined by the application in step 2.

   *swap_file*:  

   ```shell
   findmnt -no UUID -T <path_to_swap_file>
   ```

   *swap_file_offset*:

   ```shell
   filefrag -v <path_to_swap_file> | awk '{if($1="0:"){print substr($4, 1, length($4)-2)} }'
   ```

## 4.3 configure initramfs

`resume` hook should be added in `/etc/mkinitcpio.conf` after `base` and `udev`.

If `systemd` hook is used, the hibernation mechanism is already provided, and no further hooks are needed.

## 4.4 configure boot loader

resume=*swap_device*

- resume=PART_UUID
- resume="PARTLABEL=Swap Partition"

Here lists several boot loaders for demonstration. You can get PARTUUID using `blkid`

### 4.4.1 systemd boot

edit /boot/oader/entries/arch.conf

options root=UUID=... rw resume=PARTUUID=*swap_area*

### 4.4.2 grub

edit /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT="resume=PARTUUID=*swap_area*"

then regenerate the grub.cfg:

```bash
grub-mkconfig -o /boot/grub/grub.cfg
```

### 4.4.3 efibootmgr

same

resume=PARTUUID=*swap_area*

## 4.5 reboot right away

Obtain the volume's major and minor device numbers using `lsblk`, and echo them in format `major:minor` to `/sys/power/resume`.

```shell
echo <MAJ>:<MIN> > /sys/power/resume
```

If using a swap file, additionally echo the resume offset to `/sys/power/resume_offset`

```shell
echo <MAJ>:<MIN> > /sys/power/resume
echo <swap_file_offset> > /sys/power/resume_offsets
```
