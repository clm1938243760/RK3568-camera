# RK3568 Native Camera Recognition

本仓库把申请单从进入相机画面到生成结构化字段的链路迁移到 ATK-DLRK3568。
当前原生候选已在板端运行；HID、SPI 屏幕、虚拟打印机、MSC 和报告上传不在本仓库范围内。

## 当前链路

```text
IMX415 / RKISP 3840x2160 NV12
  -> V4L2 MMAP + DMA-BUF 四缓冲
  -> RGA 缩放为 256x256 BGR
  -> DocAligner FP16 RKNN 纸张检测
  -> 空闲 5 FPS、候选 15 FPS 自适应检测
  -> 四个不同帧且跨度 >= 180 ms 确认稳定
  -> 只锁定最后一张稳定帧
  -> 单次最终帧质量检查
  -> RGA 裁纸张包围框
  -> 单个组合矩阵完成拉正、旋转、长边 3200 和大区域裁剪
  -> 1175x1504 RGB 内存图
  -> PP-OCRv4 Mobile FP16 RKNN 全文 OCR，Batch 1
  -> 网页字段规则生成结构化字段
  -> 必填字段、置信度和冲突校验
```

主路径不生成 JPEG，不经过 HTTP/Base64，不采集备选图，不选图，不重拍，也不执行
第二次 OCR。质量或字段校验失败后，本轮锁定失败状态，直到纸张连续移除 0.5 秒。

## 已选模型

| 模型 | 输入 | 精度 | 用途 |
| --- | --- | --- | --- |
| DocAligner | `256x256` | FP16 RKNN | 纸张四角检测 |
| PP-OCRv4 det | `480x480` | FP16 RKNN | 大区域文字检测 |
| PP-OCRv4 rec | `48x320` | FP16 RKNN | 逐行文字识别 |

RK3568 只有一个 NPU 核心，两个 OCR 模型串行复用该核心，全部使用 Batch 1。候选
`384x480` 检测模型虽然发现更多文字框，但结构化结果出现冲突且更慢，因此未采用。

## 目录

| 路径 | 用途 |
| --- | --- |
| `native/` | V4L2、RGA、RKNN、稳定状态机、图像变换和 OCR C++ 实现 |
| `report_parser/scripts/native_structured_worker.py` | 事件驱动字段结构化 worker |
| `config/native.env.example` | 原生服务模型、设备和运行库路径 |
| `systemd/rk3568-camera-native-*.service` | 原生流水线与结构化服务 |
| `scripts/install_native_pipeline.sh` | 只安装、激活和回滚 |
| `scripts/collect_e2e_timings.py` | 不记录患者值的现场逐阶段耗时采集 |
| `docs/PORTING_PLAN.md` | 实施边界与验收门槛 |
| `docs/NATIVE_PIPELINE_RESULTS.md` | 模型 A/B 与固定图结果 |

## 部署

默认只安装文件，不切换当前服务：

```bash
sudo env \
  DOCALIGNER_RKNN_SOURCE=/tmp/docaligner_rk3568_uint8_fp16.rknn \
  PPOCR_DET_RKNN_SOURCE=/tmp/ppocrv4_det_480x480_fp16.rknn \
  NATIVE_BINARY_SOURCE=/tmp/rk3568_native_pipeline_service \
  bash scripts/install_native_pipeline.sh
```

通过板端检查后激活：

```bash
sudo env \
  DOCALIGNER_RKNN_SOURCE=/tmp/docaligner_rk3568_uint8_fp16.rknn \
  PPOCR_DET_RKNN_SOURCE=/tmp/ppocrv4_det_480x480_fp16.rknn \
  NATIVE_BINARY_SOURCE=/tmp/rk3568_native_pipeline_service \
  bash scripts/install_native_pipeline.sh --activate
```

板端从源码编译时默认单线程，避免 2 GB 内存设备并行编译 Rockchip OCR 源码时 OOM：

```bash
sudo NATIVE_BUILD_JOBS=1 bash scripts/install_native_pipeline.sh --build
```

回滚到原 JPEG 快照桥与 Python 触发器：

```bash
sudo bash scripts/install_native_pipeline.sh --rollback
```

## 运行状态

```bash
systemctl status rk3568-camera-native-pipeline.service --no-pager -l
systemctl status rk3568-camera-native-structured.service --no-pager -l
curl -sS http://127.0.0.1:8894/
```

公开状态文件为 `0644`，不含 OCR 原文和患者字段值。OCR 全文、结构化字段和固定
测试图均限制为 root `0600`，并被 Git 忽略。

## 本地检查

```powershell
python scripts/check_python37_syntax.py
python -m unittest discover -s tests -v
$env:PYTHONPATH = "report_parser/src"
python -m unittest report_parser.tests.test_native_structured_worker -v
```

C++ 测试在 RK3568 上执行：

```bash
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build -- -j1
cd native-build && ctest --output-on-failure
```
