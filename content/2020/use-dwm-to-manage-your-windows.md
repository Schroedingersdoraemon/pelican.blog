---
layout: blog
title: use dwm to manage your windows
date: 2020-09-04 22:18:56
tags:
---

# 1. introduction

Dynamic window manager, also known as dwm, is a window manager for X. It manages windows in various layouts, all of which can be applied dynamically, optimizing the environment for the application in use and the task being performed. 

- In tiled layout windows are managed in a master and stacking area. The master area contains the window which currently needs most attention, whereas the stacking area contains all other windows.

- In monocle layout all windows are maximised to the screen size.

- In floating layout windows can be resized and moved freely. Dialog windows are always managed floating, regardless of the layout applied. 

Windows are grouped by tags. Each window can be tagged with one or multiple tags. Selecting certain tags displays all windows with these tags. 

Each screen contains a small status bar which displays all available tags, the layout, the number of visible windows, the title of the focused window, and the text read from the root window name property, if the screen is focused. A floating window is indicated with an empty square and a maximised floating window is indicated with a filled square before the windows title. The selected tags are indicated with a different color. The tags of the focused window are indicated with a filled square in the top left corner. The tags which are applied to one or more windows are indicated with an empty square in the top left corner. 

dwm draws a small customizable border around windows to indicate the focus state.

# 2. features

- a single binary file, the source code is intended to never exceed 3000 SLOC.

- customized through its source code.No need to learn some weird configuration file format(like X resource files), beside **C**, to customize it for your needs. The only thing you need to learn is **C**.

# 3. download

It is *strongly* recommended to download its source code instead of any other weird distribution formats.

```bash
git clone https://git.suckless.org/dwm
```

# 4. install

```shell
cd dwm
make -j$(nproc)
doas make install
# or sudo make install
```

# 5. customization

## 5.1 patch

```shell
patch < some-certain-patch.patch
```