# SAR 投影和观测约定

ECAT 在 SAR/InSAR reader 中统一使用下面的观测约定：

```text
scalar_observation = ENU_displacement dot projection
```

这里的 `projection` 是东、北、上三个方向的投影向量。CSI 中可能存放在 `data.los`，但含义不局限于传统 LOS，也可用于 offset 或自定义标量观测。

## 默认目标正方向

| 观测类型 | 进入 CSI 后的正方向 |
| --- | --- |
| `phase_los` / `unwrapped_phase` | phase 转换后朝向卫星 |
| `los_displacement` | 朝向卫星 |
| `range_offset` | 朝向卫星 |
| `azimuth_offset` | 沿 heading 方向 |

## 为什么要区分 value 和 projection

很多产品的数值正方向和投影向量正方向可能不一样。例如外部文件可能给的是远离卫星为正的 LOS 位移，也可能给的是朝向卫星为正的投影向量。ECAT reader 的目标是把读入结果统一成上面的公式，方便后续建模和残差检查。

## 检查顺序

1. 先用 `ecat-downsample --sar-prefix ... -s` 或 YAML `-s` 画 quick-look。
2. 看原始形变图的空间范围和正负号是否符合预期。
3. 查 run metadata 中的 projection convention、target convention 和均值诊断。
4. 正式降采样前固定 `vmin/vmax` 和必要的数据过滤规则。
5. 进入反演后，用数据、synthetic、residual 三图检查符号是否一致。

## 常见误区

- GMTSAR direct-projection reader 读取 `valuefile` 和 ENU projection component grids，不是读取 azimuth/incidence 角栅格。
- `same_as_value` 表示输入 projection 的正方向和输入 value 的正方向一致；它不是“自动猜正负号”。
- azimuth offset 如果由 LOS projection 推导，需要显式说明 projection role、LOS convention 和 look side。
- quick-look 的 `coordrange` 只影响显示范围，不删除数据；真实删除点应使用 `data_filters`。

## 继续阅读

- [GAMMA SAR quick-look 与配置生成](../examples/gamma_sar_quicklook.md)
- [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [SAR Reader](../reference/sar_reader.md)
- [Downsampling App](../reference/downsampling_app.md)
