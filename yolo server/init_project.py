import logging

from utils import setup_logging
from pathlib import Path
from utils import time_it

from utils.paths import (
    YOLOSERVER_ROOT,
    CONFIGS_DIR,
    DATA_DIR,
    RUNS_DIR,
    MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    CHECKPOINTS_DIR,
    SCRIPTS_DIR,
    LOGS_DIR,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    ORIGINAL_ANNOTATIONS_DIR,
    YOLO_STAGED_LABELS_DIR
)

#配置日期记录

logger = setup_logging(base_path=LOGS_DIR, log_type="init_project", logger_name="YOLO Init Project")

@time_it(iterations=1, name="Init_Project", logger_instance=logger)
def initialize_project(logger_instance: logging.Logger = None):
    logger_instance.info("开始初始化项目".center(50, "="))