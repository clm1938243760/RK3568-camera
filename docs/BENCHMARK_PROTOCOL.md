# RK3568 与 RK3588 性能对比协议

## 固定图

两块板必须使用同一个私有 JPEG，并先核对 SHA-256。模型预热后至少执行 20 次，
只记录输入尺寸、识别区域尺寸、OCR 块数量、平均置信度和阶段耗时，不输出文字。

RK3568 原生探针：

```bash
sudo env \
  RK3568_PRIVATE_PROBE_OCR=/run/rk3568-camera/private-probe-ocr.json \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /opt/rk3568_camera/native-build/rk3568_pipeline_probe \
  /opt/rk3568_camera/models/docaligner_rk3568_uint8_fp16.rknn \
  /opt/rk3568_camera/models/ppocrv4_det_480x480_fp16.rknn \
  /userdata/aidemo/rknn_PPOCR-System_demo_native/model/ppocrv4_rec.rknn \
  /var/lib/rk3568-camera/debug-fixtures/rk3588_camera_test.jpg \
  20
```

固定图结果记录平均值；现场表记录平均、中位数、最小和最大值，不计算 P95。

## 现场全流程

启动无患者值的耗时采集器：

```bash
python3 scripts/collect_e2e_timings.py \
  --board-label rk3568 \
  --samples 20 \
  --output /tmp/rk3568-live-e2e.json
```

每个样本必须完成一次“放入申请单、结果锁定、移除申请单”。采集字段包括：

- 纸张出现到结构化结果。
- 稳定判断。
- 最终帧质量。
- RGA 裁剪。
- 组合透视与大区域裁剪。
- OCR 文字检测、行裁剪、识别预处理、NPU 推理和后处理。
- 字段结构化。
- 总时间。

## 测试条件

- 同一张纸、相同摆放区域和照明。
- 输入保持 3840x2160 NV12，标准长边 3200，识别区域 0.13 到 0.60。
- 稳定门槛为四个不同检测帧且跨度至少 180 ms。
- 最终稳定帧只处理一次；失败样本不重拍、不重试、不从统计中删除。
- 测试期间记录模型哈希、NPU 频率、CPU 负载和可用内存。
- 冷启动单独记录，不计入热流程。
- 患者图片、OCR 原文和字段值不得进入性能日志或 Git。

## 对比

```bash
python3 scripts/compare_benchmarks.py \
  --rk3588 /tmp/rk3588-live-e2e.json \
  --rk3568 /tmp/rk3568-live-e2e.json \
  --output /tmp/live-comparison.json
```

只有阶段含义、输入尺寸、模型精度和字段规则一致的行可以直接比较。
