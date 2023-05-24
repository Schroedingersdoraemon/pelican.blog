=============================
 定位手册 - 理论，实践与进阶
=============================
:date: 2023-05-21 17:18

.. contents::

1. 无线定位系统：操作，应用与实践
=================================

1.1. introduction
-----------------

  - Global Positioning System -- on the globe

  - Local Positioning System -- relative

    + Self-positioning System -- with respect to a static point

    + Remote-positioning System -- relative position of other nodes

      - Active remote positioning system -- target cooperates in positioning

      - Passive remote positioning system -- target is passive and noncooperative

1.2. basic methods
------------------

1.2.1. TOA
~~~~~~~~~~

- 所有基站与终端时间同步，小误差导致大失误
- 信号包含时间戳，提高了信号复杂度
- 基站位置已知，静态节点或者配有 gps 的动态节点

1.2.2. TDOA
~~~~~~~~~~~

- 所有基站时间同步即可（终端的石英钟往往不如基站的原子钟准确）
- 信号不需要包含时间戳，因为只需要到达时间作差

1.2.3. DOA
~~~~~~~~~~

1.2.4. RSSI
~~~~~~~~~~~

RSSI, Received Signal Strength

1.2.5. LOS.versus.NLOS
~~~~~~~~~~~~~~~~~~~~~~

TOA, TDOA, DOA 对 LOS 更加敏感。而 RSSI 受 LOS 影响较小，因为 能量-距离 关系中，
NLOS 导致的 shadowing(random) effect 可用滤波消减。因此许多 NLOS 识别、消除与
定位技术应运而生

shadowing effect
    received signal power fluctuates due to objects obstructing the propagation path

    由于传播路径上的物体遮挡，传输信号强度发生波动

1.2.6. positioning, mobility, and tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parameter estimation techniques.

例如，在 LOS/NLOS 混合环境中，往往用到卡尔曼、贝叶斯或者粒子滤波。

1.2.7. network localization.网络定位
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



2. 源定位：算法与分析
=====================
