# BA_PnP

针对BA求解PnP问题的LM算法C++语言实现，仅基于Eigen和Sophus库

编译：

```shell
mkdir build
cd build
cmake ..
make -j
```

运行：

```bash
./ba_pnp
```

TODO：

1. DLT 求解异常：求解结果接近单位阵
2. Ceres 求解异常：收敛时的误差极大
