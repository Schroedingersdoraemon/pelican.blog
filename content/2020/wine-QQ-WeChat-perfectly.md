---
layout: blog
title: "How do you perfectly run QQ and WeChat using wine?"
date: 2020-04-29 19:23:56
tags:
---

Dec 24, 2022: The official native client is released.

Aug 12, 2021: using electron-qq and electron-wechat now :)

Nov 29, 2020: 多年之后的更新： deepin.com.qq.office挺好用的，懒得折腾了。

# 1、 安装

安装 wine，下载 QQ 和微信的安装程序。

使用 wine 打开安装程序：

```bash
wine PCQQ2020.exe
```

# 2、 字体修复

可以直接将 wine 需要的字体替换为系统已知字体

## 2.1 系统中文支持

```bash
locale -a
```

如果没有以 zh_CN 开头的输出，那么请编辑 `/etc/locale-gen`，取消 zh_CN.UTF-8 前的注释

随后运行

```bash
sudo locale-gen
```

## 2.2 创建注册表规则文件

创建一个新文件，命名为 `fonts.reg`，其内容如下

```powershell
REGEDIT4

[HKEY_LOCAL_MACHINE\Software\Microsoft\WindowsNT\CurrentVersion\FontLink\SystemLink]
"待替换字体1"="现有字体"
"待替换字体2"="现有字体"
...
```

值得注意的是：现有字体应该为 `/usr/share/fonts/` 文件夹下的字体**文件名**，可用 `fc-list` 指令查询

这里提供一份个人替换文件做为参考：[chn_font.reg](/files/wine-QQ-WeChat-perfectly/chn_font.reg)

## 2.3 导入文件到注册表

以下两种方法任选其一即可

### 2.3.1

```bash
wine regedit
```

点击左上角Registry，Import Registry File，选择我们创建的文件即可。

### 2.3.2

```bash
wine regedit fonts.reg
```

我替换的现有字体为 *nerd fonts source code pro*，Arch 用户可在 archlinuxcn 源中下载 nerd-fonts-source-code-pro。
其他用户可在 nerd fonts 的 <a target="blank" href="https://github.com/ryanoasis/nerd-fonts">GitHub主页</a> 自行下载安装

您也可以更换为其它中文字体

此时打开安装包，如果字体依旧为方框，可以设定以中文环境运行wine：

```bash
env LANG=zh_CN.UTF-8 wine PCQQ2020.exe
```

此时，久违的中文字体应该就可以出现啦！

## 2.4 possible problem

至于安装微信后提示的WeChatWin.dll文件缺失问题，可以如此解决：

- 安装32位的libldap库，可以在自己发行版的包管理器中查找一下libldap，安装即可。

- 如果WeChat文件夹无有WeChatWin.dll文件，请下载此文件。

将文件放在~/.wine/drive\_c/Tencent/WeChat/文件夹下（你的微信的安装路径），此时微信也可以正常打开了

如果微信的信息框无法显示输入文字，请安装winetricks，随后执行

```bash
winetricks riched20
```

# 3、 使用方法

安装完成之后，每次打开都要输入诸如此类的指令：

```bash
env LANG=zh_CN.UTF-8 wine ~/.wine/drive_c/Tencent/WeChat/WeChat.exe
```

虽说可以将指令存为一个脚本文件然后执行，但是也是不方便，那么有没有更好的方法呢？

我的解决方法是使用 <a target="_blank" href="https://tools.suckless.org/dmenu/">dmenu</a>。

dmenu 是一个 X 下的快速、轻量级的软件启动器，它从 stdin 读取任意文本，并创建一个菜单，每一行都有一个菜单项。 然后，用户可以通过方向键或键入名称的一部分来选择一个项目，该行就会被输出到 stdout。
dmenu_run 是 dmenu 附带的 wrapper，可将其用作应用程序启动器。

既然dmenu可以创建菜单，那么可以根据此特性写一个脚本：

```bash
#!bin/sh
choices="QQ\nWeChat"
chosen=$(echo -e $choices | dmenu -p "打开程序：")
case $chosen in
    QQ)
        env LANG=zh_CN.UTF-8 wine "~/.wine/drive_c/.../QQ.exe" ;;
    WeChat)
        env LANG=zh_CN.UTF-8 wine "~/.wine/drive_c/.../WeChat.exe" ;;
esac
```

将这个脚本绑定到某个快捷键，就能完成QQ和微信的启动啦。

如果你使用的是 albert 或 ulauncher 之类，可以在 `~/.local/share/applications/` 下新建 QQ.desktop 和 WeChat.desktop 文件

例如QQ.desktop

```desktop
[Desktop Entry]
Categories=Network;InstantMessaging;
Exec=env LANG=zh_CN.UTF-8 wine "~/.wine/drive_c/.../QQ.exe所在位置"
Icon=QQ
Name=WineQQ
NoDisplay=false
StartupNotify=true
Terminal=0
Type=Application
```
