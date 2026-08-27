# RK3568 与 RK3588 性能对比协议

## 目的

将“芯片和 OCR worker 性能”与“摄像头、稳定判断和图像预处理耗时”分开测量，
避免用不同图片、不同分辨率或不同参数得出错误结论。

## 1. 固定 JPEG OCR

准备一张脱敏申请单 JPEG，并在两块板上确认 SHA-256 完全相同。

RK3588：

    python3 scripts/benchmark_ocr_http.py \
      --image /tmp/deidentified-report.jpg \
      --board-label rk3588 \
      --warmup 3 \
      --iterations 20 \
      --output /tmp/rk3588-fixed-ocr.json

RK3568 使用同样命令，只修改标签和输出文件：

    python3 scripts/benchmark_ocr_http.py \
      --image /tmp/deidentified-report.jpg \
      --board-label rk3568 \
      --warmup 3 \
      --iterations 20 \
      --output /tmp/rk3568-fixed-ocr.json

对比：

    python3 scripts/compare_benchmarks.py \
      --rk3588 /tmp/rk3588-fixed-ocr.json \
      --rk3568 /tmp/rk3568-fixed-ocr.json \
      --output /tmp/fixed-ocr-comparison.json

固定样本记录客户端总耗时、OCR 服务端耗时、OCR 框数量、中位数和 P95。

## 2. 现场全流程

在对应板上启动采集器，然后按提示完成 10 次完整放入和移除：

    python3 scripts/collect_e2e_timings.py \
      --board-label rk3568 \
      --samples 10 \
      --output /tmp/rk3568-live-e2e.json

RK3588 上使用它自己的状态和结果路径：

    python3 scripts/collect_e2e_timings.py \
      --board-label rk3588 \
      --status-file /run/rk3588-report-parser/camera-trigger.json \
      --result-file /run/rk3588-report-parser/verified-full-text.json \
      --samples 10 \
      --output /tmp/rk3588-live-e2e.json

再次使用 compare_benchmarks.py 对比两个现场结果。

## 3. 测试条件

- 同一张纸、同一摆放区域和照明。
- 两帧、0.5 秒稳定时间、3200 目标长边。
- 测试期间不运行其他 NPU 推理任务。
- 记录 NPU 当前频率、CPU 负载和可用内存。
- 第一次模型或 worker 冷启动不计入正式样本，但要单独记录。
- 失败样本不能悄悄删除，应记录错误类型后重新采集。

## 4. 结果解释

RK3568/RK3588 大于 1 表示 RK3568 更慢。例如 2.4 表示对应中位耗时约为
RK3588 的 2.4 倍。固定 JPEG 主要反映 OCR 后端，全流程还包含帧源、纸张稳定、
两帧选优、透视校正、文件轮询和调度开销。

在两类数据都完成前，不给出“RK3568 慢多少”的结论。
