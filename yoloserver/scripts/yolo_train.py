import argparse
import logging
from pathlib import Path
import sys
import ultralytics
yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0,str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1,str(utils_path))

from logging_utils import setup_logging
from config_utils import load_yaml_config,merger_configs,log_parameters
from paths import LOGS_DIR,PRETRAINED_MODELS_DIR
from system_utils import log_device_info
from datainfo_utils import log_dataset_info
from ultralytics import YOLO
def parser_args():
    parser = argparse.ArgumentParser(description="YOLO Training")
    parser.add_argument("--data",type=str,default="data.yaml",help="yaml配置文件")
    parser.add_argument("--weights",type=str,default="yolov8n.pt",help="模型权重文件")
    parser.add_argument("--batch",type=int,default=16,help="训练批次大小")
    parser.add_argument("--epochs",type=int,default=4,help="训练轮数")
    parser.add_argument("--device",type=str,default="0",help="训练设备")
    parser.add_argument("--workers",type=int,default=8,help="训练数据加载线程数")
    parser.add_argument("--use_yaml",type=bool,default=True,help="是否使用yaml配置文件")

    return parser.parse_args()

def run_training(model,yolo_args):
    results = model.train(**vars(yolo_args))
    return results

def main(logger):
    logger.info(f"YOLO工业安全生产模型训练脚本启动".center(50,"="))
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config()

        yolo_args,project_args = merger_configs(args,yaml_config)

        log_device_info()

        log_dataset_info(data_config_name="data.yaml",mode="train")

        log_parameters(project_args)

        logger.info(f"初始化yolo模型，加载模型{project_args.weights}")
        model_path = PRETRAINED_MODELS_DIR / project_args.weights
        if not model_path.exists():
            logger.warning(f"模型文件{model_path}不存在,请将模型{project_args.weights}放入")
            raise FileNotFoundError(f"模型文件{model_path}不存在")
        model = YOLO(model_path)

        run_training(model,yolo_args)

        print(yolo_args)
        print(project_args)

    except Exception as e:
        logger.error(f"加载yaml配置文件失败:{e}")
        raise e

if __name__ == "__main__":
    args = parser_args()
    logger = setup_logging(base_path=LOGS_DIR,log_type="train",model_name=args.weights.replace(".pt",""),temp_log=True)
    main(logger)