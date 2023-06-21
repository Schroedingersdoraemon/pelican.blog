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

必要的基础设施：卫星、基站、WLAN AP

少量已知的 *anchor* 或者 *reference* 节点，以及剩下的 *unlocalized* 未知节点。
未知节点不能与已知节点通信，会互相通信估算位置。

- collaborative position location
- cooperative localization
- network localization

1.3. 定位系统概览
-----------------

1.3.1. GPS
~~~~~~~~~~

27 = 24 + 3 (in case)

complete 2 rotations each 24 hours.

任何时间，任何地点，至少四颗，清晰可见！

**距离测量**
  考虑到终端机与卫星 **完美同步** (精确到纳秒级)，二者同时发出 long unique pattern
  ( pseudo random code )，根据二者信号的延迟判断距离。

  受制于成本问题，厂家往往只会在终端机里面用石英钟，无法纳秒级精确，所以额外一步。

  *syncrhonization*
    第四颗卫星用来测量终端机误差。

    The difference between the distance of the estimated receiver position from
    the fourth satellite and the pseudorange of the fourth satellite (the radius
    of the fourth satellite or the distance to the fourth satellite as measured
    by the GPS receiver) is used to calculate the error.

  另外，卫星的原子钟也周期性矫正，确保 *relativistic effect* 被消除，
  以及与陆地时钟同步。

  relativistic effect 基于可由相对论解释的两个现象：

  1. 弱重力场的时钟 tick faster
  2. 移动的时钟 tick slower

  因此，因为弱重力场，卫星时钟相对地面时钟转地更快，又因高速运动转地更慢。
  即使理论上两种效应 cancel out，最终的 net effect 是相比于地面时钟 tick faster

**卫星位置**
  终端机可存储 almanac，据此获得任何时间每个卫星的位置。

  日月重力的牵引效果被美国国防部持续观测，并将调整信息作为信号的一部分发到终端机

1.3.2. Assisted GPS
~~~~~~~~~~~~~~~~~~~

todo

GPS 的问题：Time To First Fix, TTFF 或 cold start 花费时间太久

首次开机后需要很长时间获取信号、交互数据并定位

1.3.3. INS
~~~~~~~~~~

todo

误差在积分过程中的传播被称为 *integration drift* ，

1.3.4. Integrated INS and GPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

INS 可填补两个 GPS 计算定位的空隙，也可用于 obscuration caused by maneuvering

GPS 可矫正 INS 的传播误差

1.3.5. RFID
~~~~~~~~~~~

RFID 是一个识别附着在物体上标签的无线系统，其包含一个 reader 和 RFID 标签。

根据标签种类可分为两类

- passive tags. 不含电源，适合短距离。配有被特定频率的信号激活的天线阵列。
- active RFID system. 收发系统，tag = transponder + power source.
  RFID reader 发送电磁波，tag in its vicinity 接收.
  tag modulate 电磁波，增加识别信息并发回。reader 将变频电磁波转为数字信号，

**RFID as a Positioning System**
  Received Signal Strength Indicator

  location identification based on dynamic active RFID calibration

    fixed tags serve as reference points

1.3.6. WLPS
~~~~~~~~~~~

hybrid TOA and DOA

- monitoring mobile unit (or Dynamic Base Station, DBS)
- target mobile unit ( or Active Target, or Transceiver, TRX)

DOA 用的是 DBS 上的天线阵列

1.3.7. TCAS
~~~~~~~~~~~

检测并跟踪飞行器，附近有半空碰撞风险时警告飞行员。

1.3.8. WLAN
~~~~~~~~~~~

trilateration using RSSI technique

Network Interface Card, NIC

1.3.9. vision positioning system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

todo

1.3.10. radar
~~~~~~~~~~~~~

RAdio Detection And Ranging, RADAR

1.4. Comparison
---------------

+---------+---------+---------+---------+
|         |Accuracy |LOS/NLOS |No. of BS|
+---------+---------+---------+---------+
|TOA      |Medium   |LOS      |>=3      |
+---------+---------+---------+---------+
|TDOA     |Medium   |LOS      |>=3      |
+---------+---------+---------+---------+
|DOA      |Low      |LOS      |>=2      |
+---------+---------+---------+---------+
|RSSI     |H to M   |Both     |>=3      |
+---------+---------+---------+---------+

1.5. 总结与展望
---------------


2. 源定位：算法与分析
=====================

Assuming that the disturbances in the measurements are zero-mean Gaussian
distributed, the Cramér–Rao lower bound (CRLB), which gives a lower bound
on the variance attainable by any unbiased location estimator using the same
data, will also be provided.

假定测量中的干扰是零均值高斯分布，还将提供 Cramer-Rao 下限 (CRLB)，
其给出了使用相同数据的任何无偏位置估计器可达到的方差的下限。

看完这章的学习成果：

1). 用到 TOA, TDOA, RSS 和 DOA 测量的定位算法的发展

2). 位置估计的性能测量

2.1. Introduction
-----------------

The position of a target of interest can be determined by utilizing its emitted signal
measured at an array of spatially separated receivers with a priori known locations.

**assume there are no outliers**
  shadowing and multipath propagation errors are sufficiently small

**assume LOS transmission**
  NLOS 会造成距离信息中较大的正偏差

2.2. 源定位的度量模型与原理
---------------------------

.. math::

   \textbf{r} = \textbf{f}(\textbf{x}) + \textbf{n}

where :math:`\textbf{r}` 是测量值向量，:math:`\textbf{x}` 是待定位的坐标，
:math:`\textbf{f}` 是已知的非线性函数，:math:`\textbf{n}` 是 0 均值的噪声向量。

2.2.1. TOA
~~~~~~~~~~

三个或更多基站时，把有噪的 TOA 转化为方程组，再根据已知基站 **最优化** 定位更好

令 :math:`\textbf{x} = [x \space{} y]^T` 为未知终端，
:math:`\textbf{x}_l = [x_l \space{} x_l]^T` 为已知基站，则终端与第l个基站之间的距离
:math:`d_l` 就是

.. math::

   d_l = \| \textbf{x} - \textbf{x}_l \|_2 = \sqrt{(x-x_l)^2+(y-y_l)^2}

设 :math:`t_l` 为信号从基站 *l* 到达终端机的传播时间，则显然

.. math::

   t_l = \frac{d_l}{c}

现实生活中，TOA 不可避免地存在误差，所以由 :math:`t_l` 和 *c*
相乘表示的距离与实际测量值的关系可以表示为

.. math::

   r_{TOA,l} = d_l + n_{TOA,l} = \sqrt{(x-x_l)^2+(y-y_l)^2} + n_{TOA,l}

上式可表达为向量形式

.. math::

   \textbf{r}_{TOA} = \textbf{f}_{TOA}(\textbf{x}) + \textbf{n}_{TOA}

为了便于算法的开发分析以及 CRLB 的计算，
:math:`{n_{TOA,l}}` 是零均值非相关的高斯过程。

2.2.2. TDOA
~~~~~~~~~~~

TDOA 是信号到达一对传感器的时间之差，而且 TDOA 不需要接收机的时钟和他们一起同步。将 TDOA 与传播速度 c 相乘得到接收机与两个基站的范围之差。每个 TDOA 定义了一个接收机位于其上的双曲线，而目标位置就在至少两条双曲线的交点上。

