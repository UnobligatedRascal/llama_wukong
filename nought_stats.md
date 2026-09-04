whistler@NOUGHT:/$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affi                                                                                                                                                                             nity    NUMA Affinity
GPU0     X      PIX     PHB     PHB     SYS     SYS     SYS     SYS     0-17,36-                                                                                                                                                                             53      0
GPU1    PIX      X      PHB     PHB     SYS     SYS     SYS     SYS     0-17,36-                                                                                                                                                                             53      0
GPU2    PHB     PHB      X      PIX     SYS     SYS     SYS     SYS     0-17,36-                                                                                                                                                                             53      0
GPU3    PHB     PHB     PIX      X      SYS     SYS     SYS     SYS     0-17,36-                                                                                                                                                                             53      0
GPU4    SYS     SYS     SYS     SYS      X      PIX     PHB     PHB     18-35,54                                                                                                                                                                             -71     1
GPU5    SYS     SYS     SYS     SYS     PIX      X      PHB     PHB     18-35,54                                                                                                                                                                             -71     1
GPU6    SYS     SYS     SYS     SYS     PHB     PHB      X      PIX     18-35,54                                                                                                                                                                             -71     1
GPU7    SYS     SYS     SYS     SYS     PHB     PHB     PIX      X      18-35,54                                                                                                                                                                             -71     1

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA                                                                                                                                                                              nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Hos                                                                                                                                                                             t Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the                                                                                                                                                                              CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCI                                                                                                                                                                             e Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
whistler@NOUGHT:/$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 36 37 38 39 40 41 42 43                                                                                                                                                                              44 45 46 47 48 49 50 51 52 53
node 0 size: 64345 MB
node 0 free: 21019 MB
node 1 cpus: 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 54 55 56 57 5                                                                                                                                                                             8 59 60 61 62 63 64 65 66 67 68 69 70 71
node 1 size: 64455 MB
node 1 free: 24981 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
whistler@NOUGHT:/$ lstopo
Machine (126GB total)
  Package L#0
    NUMANode L#0 (P#0 63GB)
    L3 L#0 (45MB)
      L2 L#0 (256KB) + L1d L#0 (32KB) + L1i L#0 (32KB) + Core L#0
        PU L#0 (P#0)
        PU L#1 (P#36)
      L2 L#1 (256KB) + L1d L#1 (32KB) + L1i L#1 (32KB) + Core L#1
        PU L#2 (P#1)
        PU L#3 (P#37)
      L2 L#2 (256KB) + L1d L#2 (32KB) + L1i L#2 (32KB) + Core L#2
        PU L#4 (P#2)
        PU L#5 (P#38)
      L2 L#3 (256KB) + L1d L#3 (32KB) + L1i L#3 (32KB) + Core L#3
        PU L#6 (P#3)
        PU L#7 (P#39)
      L2 L#4 (256KB) + L1d L#4 (32KB) + L1i L#4 (32KB) + Core L#4
        PU L#8 (P#4)
        PU L#9 (P#40)
      L2 L#5 (256KB) + L1d L#5 (32KB) + L1i L#5 (32KB) + Core L#5
        PU L#10 (P#5)
        PU L#11 (P#41)
      L2 L#6 (256KB) + L1d L#6 (32KB) + L1i L#6 (32KB) + Core L#6
        PU L#12 (P#6)
        PU L#13 (P#42)
      L2 L#7 (256KB) + L1d L#7 (32KB) + L1i L#7 (32KB) + Core L#7
        PU L#14 (P#7)
        PU L#15 (P#43)
      L2 L#8 (256KB) + L1d L#8 (32KB) + L1i L#8 (32KB) + Core L#8
        PU L#16 (P#8)
        PU L#17 (P#44)
      L2 L#9 (256KB) + L1d L#9 (32KB) + L1i L#9 (32KB) + Core L#9
        PU L#18 (P#9)
        PU L#19 (P#45)
      L2 L#10 (256KB) + L1d L#10 (32KB) + L1i L#10 (32KB) + Core L#10
        PU L#20 (P#10)
        PU L#21 (P#46)
      L2 L#11 (256KB) + L1d L#11 (32KB) + L1i L#11 (32KB) + Core L#11
        PU L#22 (P#11)
        PU L#23 (P#47)
      L2 L#12 (256KB) + L1d L#12 (32KB) + L1i L#12 (32KB) + Core L#12
        PU L#24 (P#12)
        PU L#25 (P#48)
      L2 L#13 (256KB) + L1d L#13 (32KB) + L1i L#13 (32KB) + Core L#13
        PU L#26 (P#13)
        PU L#27 (P#49)
      L2 L#14 (256KB) + L1d L#14 (32KB) + L1i L#14 (32KB) + Core L#14
        PU L#28 (P#14)
        PU L#29 (P#50)
      L2 L#15 (256KB) + L1d L#15 (32KB) + L1i L#15 (32KB) + Core L#15
        PU L#30 (P#15)
        PU L#31 (P#51)
      L2 L#16 (256KB) + L1d L#16 (32KB) + L1i L#16 (32KB) + Core L#16
        PU L#32 (P#16)
        PU L#33 (P#52)
      L2 L#17 (256KB) + L1d L#17 (32KB) + L1i L#17 (32KB) + Core L#17
        PU L#34 (P#17)
        PU L#35 (P#53)
    HostBridge
      PCIBridge
        PCI 01:00.0 (NVMExp)
          Block(Disk) "nvme0n1"
      PCIBridge
        PCI 02:00.0 (NVMExp)
          Block(Disk) "nvme1n1"
      PCIBridge
        PCIBridge
          PCIBridge
            PCI 08:00.0 (3D)
              CoProc(OpenCL) "opencl0d0"
          PCIBridge
            PCI 09:00.0 (3D)
              CoProc(OpenCL) "opencl0d1"
      PCIBridge
        PCIBridge
          PCIBridge
            PCI 0c:00.0 (3D)
              CoProc(OpenCL) "opencl0d2"
          PCIBridge
            PCI 0d:00.0 (3D)
              CoProc(OpenCL) "opencl0d3"
      PCIBridge
        PCI 0f:00.0 (Ethernet)
          Net "enp15s0"
      PCIBridge
        PCI 10:00.0 (Ethernet)
          Net "enp16s0"
  Package L#1
    NUMANode L#1 (P#1 63GB)
    L3 L#1 (45MB)
      L2 L#18 (256KB) + L1d L#18 (32KB) + L1i L#18 (32KB) + Core L#18
        PU L#36 (P#18)
        PU L#37 (P#54)
      L2 L#19 (256KB) + L1d L#19 (32KB) + L1i L#19 (32KB) + Core L#19
        PU L#38 (P#19)
        PU L#39 (P#55)
      L2 L#20 (256KB) + L1d L#20 (32KB) + L1i L#20 (32KB) + Core L#20
        PU L#40 (P#20)
        PU L#41 (P#56)
      L2 L#21 (256KB) + L1d L#21 (32KB) + L1i L#21 (32KB) + Core L#21
        PU L#42 (P#21)
        PU L#43 (P#57)
      L2 L#22 (256KB) + L1d L#22 (32KB) + L1i L#22 (32KB) + Core L#22
        PU L#44 (P#22)
        PU L#45 (P#58)
      L2 L#23 (256KB) + L1d L#23 (32KB) + L1i L#23 (32KB) + Core L#23
        PU L#46 (P#23)
        PU L#47 (P#59)
      L2 L#24 (256KB) + L1d L#24 (32KB) + L1i L#24 (32KB) + Core L#24
        PU L#48 (P#24)
        PU L#49 (P#60)
      L2 L#25 (256KB) + L1d L#25 (32KB) + L1i L#25 (32KB) + Core L#25
        PU L#50 (P#25)
        PU L#51 (P#61)
      L2 L#26 (256KB) + L1d L#26 (32KB) + L1i L#26 (32KB) + Core L#26
        PU L#52 (P#26)
        PU L#53 (P#62)
      L2 L#27 (256KB) + L1d L#27 (32KB) + L1i L#27 (32KB) + Core L#27
        PU L#54 (P#27)
        PU L#55 (P#63)
      L2 L#28 (256KB) + L1d L#28 (32KB) + L1i L#28 (32KB) + Core L#28
        PU L#56 (P#28)
        PU L#57 (P#64)
      L2 L#29 (256KB) + L1d L#29 (32KB) + L1i L#29 (32KB) + Core L#29
        PU L#58 (P#29)
        PU L#59 (P#65)
      L2 L#30 (256KB) + L1d L#30 (32KB) + L1i L#30 (32KB) + Core L#30
        PU L#60 (P#30)
        PU L#61 (P#66)
      L2 L#31 (256KB) + L1d L#31 (32KB) + L1i L#31 (32KB) + Core L#31
        PU L#62 (P#31)
        PU L#63 (P#67)
      L2 L#32 (256KB) + L1d L#32 (32KB) + L1i L#32 (32KB) + Core L#32
        PU L#64 (P#32)
        PU L#65 (P#68)
      L2 L#33 (256KB) + L1d L#33 (32KB) + L1i L#33 (32KB) + Core L#33
        PU L#66 (P#33)
        PU L#67 (P#69)
      L2 L#34 (256KB) + L1d L#34 (32KB) + L1i L#34 (32KB) + Core L#34
        PU L#68 (P#34)
        PU L#69 (P#70)
      L2 L#35 (256KB) + L1d L#35 (32KB) + L1i L#35 (32KB) + Core L#35
        PU L#70 (P#35)
        PU L#71 (P#71)
    HostBridge
      PCIBridge
        PCIBridge
          PCIBridge
            PCI 86:00.0 (3D)
              CoProc(OpenCL) "opencl0d4"
          PCIBridge
            PCI 87:00.0 (3D)
              CoProc(OpenCL) "opencl0d5"
      PCIBridge
        PCIBridge
          PCIBridge
            PCI 8b:00.0 (3D)
              CoProc(OpenCL) "opencl0d6"
          PCIBridge
            PCI 8c:00.0 (3D)
              CoProc(OpenCL) "opencl0d7"