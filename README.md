# RK3568 Camera Recognition Port

这是从 RK3588 相机识别链路拆出的独立 RK3568 迁移工作仓库。当前目标是先在
ATK-DLRK3568 上复现相同的纸张检测、两帧选优、透视校正和全文 OCR，再用同一
样本和同一参数测出与 RK3588 的真实耗时差。

当前状态是“已完成板端依赖准备，等待 RK3568 真机识别验证”，不是已经完成性能验收。

## 基线来源

- 上层识别代码来自 RK3588-camera 提交
  795c0c76e3ea065366f0478e47862cb4704326a3。
- DocAligner 模型及其 SHA-256 保持不变，在 RK3568 上使用 ONNX Runtime CPU。
- RK3568 已有 rk3568-ppocr.service，HTTP 契约为
  POST http://127.0.0.1:5002/ocr。
- 已知帧服务为 rk3568-patient-frame.service，默认接口为
  http://127.0.0.1:8090/api/frame.jpg?quality=95。
- 预检已确认 `/dev/video9` 和 `/dev/video10` 属于 USB Composite Camera，
  现有帧服务实际使用 `/dev/video9`，输出 1280x720 MJPEG，约 25.74 FPS。

## 数据链路

    RK3568 帧服务 :8090
      -> HTTP JPEG 快照桥接，5 FPS 原子轮转文件
      -> DocAligner ONNX Runtime CPU 纸张检测
      -> 稳定 0.5 秒
      -> 连续采集 2 帧并选择最佳画面
      -> 透视校正，目标长边 3200
      -> RK3568 PP-OCR :5002，串行单 NPU
      -> schema-v2 全文结果
      -> 8894 监看页和无敏感字段的耗时记录

首轮参数故意与 RK3588 一致，用于公平比较。得到基线后，才能单独测试
1920 长边、ROI 缩小或降低采集分辨率等优化。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| camera_ocr_overlay.py | RK3568 品牌和路径的 8894 监看服务 |
| report_parser/ | DocAligner、两帧选优、透视校正和全文 OCR 上层代码 |
| scripts/rk3568_snapshot_bridge.py | 将现有 HTTP JPEG 转成识别器所需的轮转快照 |
| scripts/benchmark_ocr_http.py | 同一脱敏 JPEG 的 PP-OCR 固定样本基准 |
| scripts/collect_e2e_timings.py | 纸张出现到结果生成的现场全流程采样 |
| scripts/compare_benchmarks.py | RK3588 与 RK3568 结果对比 |
| scripts/rk3568_preflight.sh | 只读检查摄像头、NPU、运行库和服务 |
| systemd/ | 三个独立的 RK3568 相机识别服务 |

解析器内部 Python 包名暂时保留 rk3588_report_parser，以保持行为和测试
基线不变；它只是兼容命名，不代表调用 RK3588 专用运行库。

## 安全部署

先执行只读预检：

    bash scripts/rk3568_preflight.sh

安装文件但不启用、不启动服务：

    sudo apt-get install -y python3-opencv python3-numpy
    sudo bash install.sh --bootstrap-python

确认 /var/lib/rk3568-camera/camera.env 中的帧接口、方向和裁剪范围后再激活：

    sudo bash install.sh --activate

安装器不会修改或重启既有的帧服务、PP-OCR、网关、USB gadget、打印或
MSC 服务。监看页使用 http://BOARD_IP:8894/，不会占用 RK3588 的 8893。
Debian 的 OpenCV 3.2 负责 JPEG 和图像处理；固定版本的 ONNX Runtime 1.14.0
负责 DocAligner 推理，因此不依赖 OpenCV 的 ONNX DNN 支持。

## 性能比较

固定样本和现场全流程必须分别测量。完整命令见
[基准协议](docs/BENCHMARK_PROTOCOL.md)，迁移顺序见
[迁移计划](docs/PORTING_PLAN.md)。

## 本地检查

    python -m py_compile camera_ocr_overlay.py scripts\*.py
    python scripts\check_python37_syntax.py
    python -m unittest discover -s tests -v

    $env:PYTHONPATH = "report_parser/src"
    python -m unittest discover -s report_parser/tests -v
