build flutter on arm linux
##########################
:date: 2026-02-26 21:40

.. contents::

The entire structure of flutter is a 3-layer system:

+----------------------------------------------+
|      flutter 3 layer system                  |
+==============================================+
|| flutter CLI (Dart)       ←  ./bin/flutter   |
||        flutter_tools.snapshot               |
+----------------------------------------------+
|| Flutter Engine (C++/Skia)   ← gclient / gn  |
|| prebuilt binaries                           |
+----------------------------------------------+
||             System Toolchain                |
||         (clang, ninja, dart SDK)            |
+----------------------------------------------+

Flutter bootstraps without further ado, for the ./bin/flutter is a wrapper,
which means simply execute

.. code-block:: bash
    
    ./bin/flutter

is sufficient enought to run flutter on arm linux.

|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|

The following text is deprecated and hence omitted.

1. intro
--------
This January I flashed my macbook air m1 to gentoo linux, and dozens of tools
have to be built on my own.

Especially the official sdk is solely provided in amd64, and google series tools
seem so utterly annoying.

1. prerequisites
----------------

.. code-block:: bash
    
    git clone --depth=1 https://github.com/flutter/flutter
    git clone --depth=1 https://chromium.googlesource.com/chromium/tools/depot_tools

depot_tools is needed and should be **prefixed**, so update PATH

.. code-block:: bash
    
    export PATH=$(pwd)/depot_tools:$PATH

then prepare for the gclient bootstrap

.. code-block:: bash

    cd flutter/
    cp engine/scripts/standard.gclient .gclient
    gclient sync --verbose

I failed in java part so I manually installed openjdk-bin:21.
Also openjdk part in DEPS is commented out.





