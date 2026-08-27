# RK3568 迁移计划

## 迁移边界

本仓库只迁移摄像头纸张识别链路，不接管 RK3568 现有网关、HID、虚拟打印机、
报告上传、MSC 或 USB gadget。原有 rk3568_patient_ocr_server 也保持独立。

## 已确认条件

| 项目 | 当前证据 | 处理方式 |
| --- | --- | --- |
| 系统 | Debian 10、Linux 4.19.232、Python 3.7.3 | 生产子集保持 Python 3.7 语法和 API 兼容 |
| NPU | fde40000.npu | 只调用现有 RK3568 PP-OCR，不复制 RK3588 runtime |
| OCR | rk3568-ppocr.service，端口 5002 | 保持 /health 和 /ocr 协议 |
| 图像 | 已有 8090 HTTP JPEG，当前记录为 /dev/video9 1280x720 MJPEG | 用快照桥接隔离具体 V4L2 节点 |
| 纸张检测 | DocAligner ONNX | RK3568 使用 OpenCV DNN CPU |

## 不允许直接复制

- RK3588 的 .rknn 模型。
- RK3588 的 librknnrt.so 或 RGA 运行库。
- RK3588 三 NPU 核心并行 worker。
- /dev/video22、/dev/video23、fdab0000.npu 等 RK3588 路径。

## 执行阶段

### A. 只读预检

运行 scripts/rk3568_preflight.sh，确认实际摄像头节点、格式、分辨率、NPU
频率节点、OpenCV/Numpy/Pillow 和两个现有 HTTP 服务。此阶段不启动新服务。

### B. 固定样本基准

在 RK3588 和 RK3568 使用同一个脱敏 JPEG，先预热 3 次，再各执行 20 次 OCR。
只有 SHA-256 相同的结果允许比较。它隔离 NPU、worker 和 HTTP 包装层性能。

### C. 实时图像接入

先单独启动 rk3568-camera-snapshot-bridge.service，确认轮转 JPEG 完整、帧率
稳定且画面确实来自纸张摄像头。若 /dev/video9 不是目标摄像头，只修改现有
帧服务配置或 FRAME_ENDPOINT，不在识别业务代码中硬编码新节点。

### D. 识别链路

启动纸张触发服务和 8894 监看服务。首轮固定使用：

    stable_seconds = 0.5
    burst_frames = 2
    ocr_document_long_side = 3200
    ocr_tiling = disabled
    ocr_refinement_max_regions = 0

连续完成至少 10 次“放入纸张、生成结果、移除纸张”循环，记录检测、两帧采集、
主 OCR 和总耗时。

### E. 优化实验

基线完成后一次只改变一个变量，优先顺序：

1. 长边从 3200 降至 1920。
2. 使用已配置字段 ROI，减少送入 OCR 的区域。
3. 调整摄像头原始分辨率和 JPEG 品质。
4. 在不降低字段可靠性的前提下调整稳定时间。

每次实验必须保留样本数量、输入 SHA、字段完整性和置信度结果。

## 完成标准

- 纸张框出现、移除和重入状态正确。
- 同一张申请单不会先要求移除后又继续录入。
- 结构化字段、姓名和患者 ID 与 RK3588 行为一致。
- 固定样本至少 20 次、现场流程至少 10 次。
- 输出 RK3568/RK3588 中位数、P95、差值和倍数。
- 不提交患者图片、OCR 全文、数据库、密钥或医院接口配置。
