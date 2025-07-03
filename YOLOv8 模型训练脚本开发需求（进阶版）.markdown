# YOLOv8 模型训练脚本开发需求（进阶版）

## 文档信息
- **文档编号**：REQ-YOLO-TRAIN-002
- **版本**：1.3
- **作者**：雨霓
- **创建日期**：2025年6月11日
- **更新日期**：2025年6月11日 11:35 AM EDT
- **状态**：草稿
- **审核人**：待定
- **分发范围**：开发团队、测试团队、项目经理

## 1. 项目背景
本项目为安全帽检测系统的一部分，基于 Ultralytics YOLOv8/YOLOv11，开发进阶版模型训练脚本（`yolo_train.py`），提升可复现性、健壮性和实用性。基础脚本已实现模型加载、训练和结果保存（REQ-YOLO-TRAIN-001）。日志功能（初始化、内容、重命名）已实现，本需求聚焦参数管理、设备/数据集信息记录、模型拷贝等功能，确保实验可追溯、审计和调试便捷。

## 2. 目标
- **业务目标**：
  - 提供详细训练记录（参数来源、设备、数据集、结果），支持实验可复现性和审计。
  - 生成标准化日志和结果目录，简化调试和分析。
- **技术目标**：
  - 完善日志系统，集成参数管理（加载、生成、合并）。
  - 记录设备和数据集信息，拷贝预训练模型，提升健壮性。
  - 模块化设计，支持未来扩展（如 COCO 格式数据集）。
- **交付目标**：
  - 交付增强版 `yolo_train.py` 和工具模块，优先完成参数相关功能。
  - 更新 README 和测试报告，符合企业规范。

## 3. 功能模块
### 3.1 功能模块概览
以下为功能模块清单，按优先级排序（高：必须实现；中：时间允许实现；低：可选）：

| 功能编号 | 模块名称                   | 优先级 | 状态     | 开发理由                                                                 | 依赖模块                     |
|----------|---------------------------|--------|----------|-------------------------------------------------------------------------|-----------------------------|
| FR-000   | 日志功能（初始化+内容）    | 高     | 已实现   | 核心记录工具，确保训练过程可追溯，满足审计和调试需求                     | 无                          |
| FR-001   | 日志重命名（rename_log_file） | 高     | 未实现 | 标准化日志命名，便于管理多轮实验                                         | FR-000                     |
| FR-002   | 加载 YAML 配置（load_yaml_config） | 高     | 未实现   | 加载配置文件，确保参数可追溯，减少配置错误                               | 无                          |
| FR-003   | 生成默认 YAML（generate_default_yaml） | 高     | 未实现   | 配置文件缺失时生成默认值，提升脚本健壮性和用户体验                       | FR-002                     |
| FR-004   | 参数合并（merge_configs）  | 高     | 未实现   | 合并 CLI、YAML、默认参数，确保灵活性和一致性                             | FR-002, FR-003             |
| FR-005   | 设备信息（system_utils.py） | 中     | 未实现   | 记录硬件环境（如 GPU、PyTorch 版本），支持实验复现和调试                | FR-000                     |
| FR-006   | 数据集信息（dataset_utils.py） | 中     | 未实现   | 记录数据集信息（如类别数、样本数），确保数据一致性和审计                | FR-000                     |
| FR-007   | 模型输出信息（results_utils.py） | 中     | 未实现   | 模型训练结果信息               | FR-000, FR-005, FR-006     |
| FR-008   | 模型拷贝（copy_checkpoint_models） | 中     | 未实现   | 拷贝预训练模型到结果目录，便于版本管理和复现                           | 无                          |
| FR-009   | 主训练脚本（日志内容、错误处理、控制台输出） | 低     | 部分实现 | 整合训练逻辑、错误处理和进度反馈，提升健壮性和用户体验，非核心           | FR-000, FR-002-008         |

### 3.2 功能模块详情
#### FR-000：日志功能（初始化+内容）（优先级：高，已实现）
- **开发理由**：
  - 日志是训练过程的核心记录工具，保存参数、设备、结果等信息，确保实验可复现性和审计性。
  - 标准化日志初始化（如 `temp-YYYYMMDD-HHMMSS-yolov8n.log`）为其他模块（如 FR-001、FR-007）提供基础。
- **实现思路**（已实现）：
  - 调用 `setup_logging(base_path, log_type='train', model_name='yolov8n', encoding='utf-8-sig')`。
  - 使用 `logging` 模块，配置 INFO 级别，格式：`时间 - 模块 - 消息`。
  - 保存日志到 `base_path/logging/train`，文件名基于时间戳和模型名（如 `temp-20250611-113500-yolov8n.log`）。
  - 示例：
    ```python
    import logging
    from pathlib import Path
    def setup_logging(base_path, log_type='train', model_name='yolov8n', encoding='utf-8-sig'):
        log_dir = Path(base_path) / 'logging' / log_type
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f'temp-{datetime.now().strftime("%Y%m%d-%H%M%S")}-{model_name}.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            encoding=encoding
        )
    ```

#### FR-001：日志重命名（rename_log_file）（优先级：高，已实现）
- **开发理由**：
  - 标准化日志命名（如 `trainN-YYYYMMDD_HHMMSS-yolov8n.log`）便于管理多轮实验，避免文件冲突。
  - 递增编号（`trainN`）支持实验区分，满足企业级日志管理规范。
- **实现思路**（已实现）：
  - 调用 `rename_log_file`，扫描 `logs/train` 目录，获取最大编号 `N`。
  - 重命名临时日志（如 `temp-20250611-113500-yolov8n.log` → `train1-20250611_113500-yolov8n.log`）。
  - 示例：
    ```python
    from pathlib import Path
    def rename_log_file(temp_log_path: Path, model_name: str) -> Path:
        log_dir = temp_log_path.parent
        existing_logs = [f.name for f in log_dir.glob('train*.log')]
        n = max([int(f.split('-')[0].replace('train', '')) for f in existing_logs if f.startswith('train')] or [0]) + 1
        new_name = log_dir / f'train{n}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-{model_name}.log'
        temp_log_path.rename(new_name)
        logging.info(f'日志重命名: {new_name}')
        return new_name
    ```

#### FR-002：加载 YAML 配置（load_yaml_config）（优先级：高）
- **开发理由**：
  - 加载 `data.yaml` 配置文件提供训练参数（如 `epochs`, `batch`），是参数管理的核心。
  - 确保参数可追溯，减少配置错误，支持实验复现。
- **实现思路**：
  - 使用 `pyyaml` 加载 `data.yaml`，解析字段（如 `train`, `val`, `nc`, `names`）。
  - 检查文件存在性和格式，抛出异常（如 `FileNotFoundError`）并记录日志。
  - 返回配置字典，供后续合并（FR-004）使用。
  - 示例：
    ```python
    import yaml
    from pathlib import Path
    def load_yaml_config(yaml_path: str) -> dict:
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logging.info(f'加载配置文件: {yaml_path}')
            return config
        except FileNotFoundError:
            logging.error(f'配置文件不存在: {yaml_path}')
            raise
        except yaml.YAMLError as e:
            logging.error(f'YAML 解析错误: {e}')
            raise
    ```

#### FR-003：生成默认 YAML（generate_default_yaml）（优先级：高）
- **开发理由**：
  - 配置文件缺失时生成默认 YAML，提升脚本健壮性和用户体验。
  - 提供默认参数（如 `epochs=5`, `batch=16`），确保训练可继续。
- **实现思路**：
  - 检查 `data.yaml` 是否存在，若不存在则生成默认配置文件。
  - 默认参数：`train: images/train`, `val: images/val`, `nc: 0`, `names: []`, `epochs: 5`, `batch: 16`。
  - 保存到指定路径（如 `configs/data.yaml`），记录日志。
  - 示例：
    ```python
    import yaml
    from pathlib import Path
    def generate_default_yaml(yaml_path: str) -> dict:
        default_config = {
            'train': 'images/train',
            'val': 'images/val',
            'nc': 0,
            'names': [],
            'epochs': 5,
            'batch': 16
        }
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(default_config, f)
        logging.info(f'生成默认配置文件: {yaml_path}')
        return default_config
    ```

#### FR-004：参数合并（merge_configs）（优先级：高）
- **开发理由**：
  - 合并命令行（CLI）、YAML 和默认参数（优先级：CLI > YAML > 默认值），确保配置灵活性和一致性。
  - 避免参数冲突，提升用户体验，满足可复现性要求。
- **实现思路**：
  - 使用 `argparse` 解析 CLI 参数（如 `--epochs`, `--data`）。
  - 调用 `load_yaml_config`（FR-002）获取 YAML 参数，若失败则调用 `generate_default_yaml`（FR-003）。
  - 合并参数：CLI 参数覆盖 YAML，YAML 覆盖默认值。
  - 记录合并后的参数及来源到日志（依赖 FR-000）。
  - 示例：
    ```python
    import argparse
    def merge_configs() -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--data', type=str, default='configs/data.yaml')
        args = parser.parse_args()
        config = generate_default_yaml(args.data) if not Path(args.data).exists() else load_yaml_config(args.data)
        final_config = {'epochs': 5, 'batch': 16}  # 默认值
        final_config.update(config)  # YAML 覆盖默认值
        final_config.update({k: v for k, v in vars(args).items() if v is not None})  # CLI 覆盖
        for k, v in final_config.items():
            source = 'CLI' if k in vars(args) and vars(args)[k] is not None else 'YAML' if k in config else '默认值'
            logging.info(f'- {k}: {v} (来源: {source})')
        return final_config
    ```

#### FR-005：设备信息（system_utils.py）（优先级：中）
- **开发理由**：
  - 记录硬件环境（如 OS、CPU、GPU、PyTorch 版本）支持实验复现，方便调试硬件相关问题。
  - 满足审计需求，确保环境一致性。
- **实现思路**：
  - 实现 `system_utils.get_device_info`，使用 `psutil` 获取 CPU/内存，`torch.cuda` 获取 GPU 信息。
  - 输出 JSON 格式，记录到日志（依赖 FR-000）。
  - 处理异常（如 `nvidia-smi` 不可用），回退到 CPU 模式。
  - 示例：
    ```python
    # utils/system_utils.py
    import psutil
    import torch
    import platform
    import json
    def get_device_info() -> dict:
        try:
            info = {
                'OS': {'Type': platform.system(), 'Version': platform.version()},
                'CPU': {'Cores': psutil.cpu_count(), 'Usage': psutil.cpu_percent()},
                'Memory': {'Total': f'{psutil.virtual_memory().total / 1e9:.2f} GB'},
                'GPU': {'Available': torch.cuda.is_available(), 'Count': torch.cuda.device_count()}
            }
            if info['GPU']['Available']:
                info['GPU']['Model'] = torch.cuda.get_device_name(0)
            logging.info(f'设备信息: {json.dumps(info, indent=2)}')
            return info
        except Exception as e:
            logging.warning(f'设备信息获取失败: {e}')
            return {'Errors': [str(e)]}
    ```

#### FR-006：数据集信息（dataset_utils.py）（优先级：中）
- **开发理由**：
  - 记录数据集信息（如类别数、样本数）确保数据一致性，支持实验复现和审计。
  - 验证数据集配置正确，减少训练错误。
- **实现思路**：
  - 实现 `dataset_utils.get_dataset_info`，解析 `data.yaml`（依赖 FR-002）。
  - 使用 `pathlib.Path.glob` 统计 `train`/`val` 目录的样本数（`.jpg`, `.png`）。
  - 记录到日志（依赖 FR-000）。
  - 示例：
    ```python
    # utils/dataset_utils.py
    from pathlib import Path
    import yaml
    def get_dataset_info(yaml_path: str) -> dict:
        config = load_yaml_config(yaml_path)
        train_path = Path(config.get('train', 'images/train'))
        val_path = Path(config.get('val', 'images/val'))
        info = {
            'Config': yaml_path,
            'Classes': config.get('nc', 0),
            'Names': config.get('names', []),
            'Train_Samples': len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png'))),
            'Val_Samples': len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
        }
        logging.info(f'数据集信息: {info}')
        return info
    ```

#### FR-007：模型结果信息（log_training_info）（优先级：中）
- **开发理由**：
  - 整合设备（FR-005）、数据集（FR-006）和参数（FR-004）信息到日志，确保训练过程记录完整。
  - 支持实验复现和审计，简化调试。
- **实现思路**：
  - 实现 `log_training_info`，调用 `get_device_info`（FR-005）、`get_dataset_info`（FR-006）和 `merge_configs`（FR-004）。
  - 格式化信息，记录到日志（依赖 FR-000）。
  - 示例：
    ```python
    def log_training_info(config: dict, device_info: dict, dataset_info: dict):
        logging.info('===== 训练信息 =====')
        logging.info(f'参数: {config}')
        logging.info(f'设备: {json.dumps(device_info, indent=2)}')
        logging.info(f'数据集: {dataset_info}')
    ```

#### FR-008：模型拷贝（copy_checkpoint_models）（优先级：中）
- **开发理由**：
  - 拷贝预训练模型到结果目录（如 `runs/detect/trainN/weights`）确保实验完整性，便于版本管理和复现。
  - 支持审计，记录使用的模型文件。
- **实现思路**：
  - 实现 `copy_checkpoint_models`，获取 `--weights` 参数，拷贝到 `runs/detect/trainN/weights`。
  - 使用 `shutil.copy` 和 `pathlib.Path` 确保安全操作。
  - 示例：
    ```python
    from pathlib import Path
    import shutil
    def copy_checkpoint_models(weights: str, run_dir: str):
        target_dir = Path(run_dir) / 'weights'
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(weights).name
        shutil.copy(weights, target_path)
        logging.info(f'模型拷贝: {weights} -> {target_path}')
    ```

#### FR-009：主训练脚本（日志内容、错误处理、控制台输出）（优先级：低）
- **开发理由**：
  - 整合训练逻辑，记录详细日志内容（训练结果、耗时等），捕获异常（如 CUDA 错误），提供进度反馈。
  - 提升健壮性和用户体验，非核心功能可部分实现。
- **实现思路**：
  - **日志内容**：记录训练开始/结束时间、结果（mAP@50等）、耗时，依赖 FR-000。
  - **错误处理**：使用 `try-except` 捕获异常（如 `torch.cuda.OutOfMemoryError`, `UnicodeEncodeError`），记录到日志。
  - **控制台输出**：打印每轮损失/mAP和最终结果，时间紧张可简化。
  - 示例：
    ```python
    from ultralytics import YOLO
    import time
    try:
        model = YOLO('yolov8n.pt')
        start_time = time.time()
        logging.info('训练开始')
        results = model.train(data='data.yaml', epochs=10)
        logging.info(f'训练结果: mAP@50={results.box.map50}, mAP@50:95={results.box.map}')
        logging.info(f'耗时: {time.time() - start_time:.2f}秒')
        print(f'最终结果: mAP@50={results.box.map50}')
    except Exception as e:
        logging.error(f'训练失败: {e}\n{traceback.format_exc()}')
        sys.exit(1)
    ```

## 4. 非功能需求
- **NF-001：代码规范**：
  - 遵循 PEP 8，包含文档字符串，使用类型提示（如 `def load_yaml_config(yaml_path: str) -> dict`）。
- **NF-002：兼容性**：
  - 支持 Windows/Linux，GPU/CPU，YOLOv8/YOLOv11。
  - 处理中文路径（`utf-8-sig` 或 `utf-8` 编码）。
- **NF-003：性能**：
  - 日志记录不显著影响训练（如避免频繁 I/O）。
- **NF-004：可维护性**：
  - 模块化设计，分离 `system_utils.py`, `dataset_utils.py`, `logging_utils.py`。
- **NF-005：文档**：
  - 更新 README：新功能、使用示例（如 `python yolo_train.py --data data.yaml --epochs 10`）、依赖安装。
  - 测试报告：日志样本、结果目录（如 `runs/detect/train1`）、指标截图。
- **NF-006：安全性**：
  - 日志不包含敏感信息（如密码、API 密钥），检查 `os.environ` 和参数。
  - 使用 `pathlib` 确保安全文件操作。

## 5. 技术要求
- **编程语言**：Python 3.12+
- **依赖库**：
  - `ultralytics>=8.2.0`, `pyyaml>=6.0`, `psutil>=5.9.0`, `torch>=2.8.0`
  - 标准库：`logging`, `pathlib`, `argparse`, `shutil`
- **数据集**：YOLO 格式 `data.yaml`（如 `train: images/train, val: images/val, nc: 2, names: ['helmet', 'no_helmet']`）
- **模型**：`yolov8n.pt` 或训练后模型
- **硬件**：推荐 GPU（8GB，如 RTX 3070），最低 CPU（8核，16GB内存）
- **项目结构**：
  ```
  project/
  ├── yolo_train.py
  ├── utils/
  │   ├── system_utils.py
  │   ├── dataset_utils.py
  │   ├── logging_utils.py
  ├── configs/
  │   └── data.yaml
  ├── logs/
  │   └── train/
  ├── runs/
  │   └── detect/
  └── pretrained_models/
      └── yolov8n.pt
  ```
- **版本控制**：Git 分支 `feature/train-advanced`，提交格式如 `feat: add yaml config loading`.

## 6. 验收标准
- **AC-001：日志功能**（高）：
  - 日志初始化为 `temp-YYYYMMDD-HHMMSS-yolov8n.log`，包含训练信息。
- **AC-002：日志重命名**（高）：
  - 重命名为 `trainN-YYYYMMDD_HHMMSS-yolov8n.log`，编号唯一。
- **AC-003：加载 YAML 配置**（高）：
  - 正确加载 `data.yaml`，记录路径和参数。
- **AC-004：生成默认 YAML**（高）：
  - 配置文件缺失时生成默认值（如 `epochs: 5`）。
- **AC-005：参数合并**（高）：
  - CLI 覆盖 YAML，YAML 覆盖默认值，记录来源。
- **AC-006：设备信息**（中）：
  - 日志包含 JSON 格式设备信息（OS、CPU、GPU）。
- **AC-007：数据集信息**（中）：
  - 日志记录类别数、样本数（如 `训练样本: 1000`）。
- **AC-008：组合信息**（中）：
  - 日志整合设备、数据集、参数信息。
- **AC-009：模型拷贝**（中）：
  - 模型拷贝至 `runs/detect/trainN/weights`。
- **AC-010：主训练脚本**（低）：
  - 记录训练结果、耗时，捕获异常，输出进度。

## 7. 开发流程
- **流程**：
  1. **需求分析**（1天，6月11日）：
     - 确认模块优先级，聚焦高优先级（FR-000-004）。
  2. **代码开发**（4天，6月12-15日）：
     - 完善日志功能，开发 FR-002-004（参数管理）。
     - 实现 FR-005-008（设备、数据集、模型拷贝），视时间实现 FR-009。
  3. **功能测试**（2天，6月16-17日）：
     - 验证日志、参数、结果目录，检查 Windows/Linux、GPU/CPU。
  4. **文档编写**（1天，6月18日）：
     - 更新 README，提交测试报告。
  5. **提交审核**（1天，6月19日）：
     - 提交 Pull Request，配合审查。
- **时间表**：6月11-19日（8个工作日，含1天缓冲）。
- **职责**：
  - 开发：实现代码、文档。
  - 测试：验证功能、日志、结果。
  - 经理：审核进度。
  - 审核人：检查代码、文档。

## 8. 风险与缓解措施
- **R-001：日志重命名冲突**：
  - 检查 `logs/train`，动态分配编号。
- **R-002：YAML 文件错误**：
  - 提供默认值，记录错误日志。
- **R-003：硬件信息获取失败**：
  - 回退 CPU 模式，记录警告。
- **R-004：中文路径编码问题**：
  - 使用 `utf-8` 或 `utf-8-sig`，捕获 `UnicodeEncodeError`。
- **R-005：参数合并错误**：
  - 验证 CLI > YAML > 默认值逻辑。
- **R-006：进度延误**：
  - 优先 FR-000-004，推迟 FR-009。

## 9. 交付物
- **代码**：
  - `yolo_train.py`, `utils/system_utils.py`, `utils/dataset_utils.py`, `utils/logging_utils.py`.
- **文档**：
  - README：新功能、使用示例、依赖安装。
  - 测试报告：日志样本、结果目录、指标截图。
- **版本控制**：
  - 提交至 `feature/train-advanced`，示例：`feat: add yaml config loading`.

## 10. 参考资源
- **代码**：现有 `yolo_train.py`, `utils/*.py`.
- **文档**：Ultralytics 文档（https://docs.ultralytics.com）。
- **数据集**：`configs/data.yaml`（安全帽检测，1000训练+200验证）。
- **模型**：`pretrained_models/yolov8n.pt`.
- **环境**：Windows 11/Ubuntu 22.04，Python 3.12，RTX 3070。

## 11. 审批流程
1. 提交 Pull Request 至 `feature/train-advanced`。
2. 审核代码（PEP 8、功能实现）、文档、测试报告。
3. 测试团队验证日志、结果、指标。
4. 经理批准合并至 `main`。
5. 存档文档至项目库。