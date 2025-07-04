import sys
from pathlib import Path
import logging
import argparse

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0,str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(0,str(utils_path))


from data_validation import (
    verify_dataset_config,
    verify_split_uniqueness,
    delete_invalid_files
)

from logging_utils import setup_logging
from paths import LOGS_DIR, CONFIGS_DIR

DEFAULT_SAMPLE_RATIO = 0.1
DEFAULT_MIN_SAMPLES = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolo数据集验证工具")
    parser.add_argument(
        '--mode','-m',
        type=str,
        default='FULL',
        choices=['SAMPLE','FULL'],
        help='验证模式,SAMPLE表示只验证样本数量,FULL表示完整验证'
    )

    parser.add_argument(
        '--task','-t',
        type=str,
        default='detection',
        choices=['detection','segmentation'],
        help='任务类型,检测(detection)或分割(segmentation)'
    )

    parser.add_argument(
        '--delete-invalid','-d',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否在验证失败以后提供删除不合法图像与标签的选项,默认开启,使用 --no-delete-invalid 明确禁用'
    )

    parser.add_argument(
        '--sample-ratio','-r',
        type=float,
        default=DEFAULT_SAMPLE_RATIO,
        help=f'样本采样比例,默认为{DEFAULT_SAMPLE_RATIO},用于SAMPLE模式'
    )

    parser.add_argument(
        '--min-samples','-n',
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f'样本数量下限,默认为{DEFAULT_MIN_SAMPLES},用于SAMPLE模式'
    )
    args = parser.parse_args()
    ENABLE_DELETE_INVALID = args.delete_invalid

    logger = setup_logging(base_path=LOGS_DIR,log_type="dataset_verify")
    logger.info(f"当前验证配置:模式={args.mode},任务类型={args.task},删除非法数据={args.delete_invalid}"
                f"采样比例={args.sample_ratio},最小样本数量={args.min_samples}")

    logger.info(f"开始数据集配置,内容验证,与类别分布分析(模式:{args.mode})")
    basic_validation_passed_initial,invalid_data_list,all_image_paths_from_validation = (
        verify_dataset_config(
            yaml_path=CONFIGS_DIR / "data.yaml",
            mode=args.mode,
            task_type=args.task,
            sample_ratio=args.sample_ratio,
            min_samples=args.min_samples
        )
    )
    basic_validation_problems_handled = basic_validation_passed_initial

    if not basic_validation_passed_initial:
        logger.error('基础数据集配置验证失败,请检查日志文件')
        logger.error(f"检测到{len(invalid_data_list)}个不合法的图像标签对,详细信息如下:")
        for i,item in enumerate(invalid_data_list):
            logger.error(f"不合法数据{i+1} 图像: {item['image_path']} 标签:"
                         f" {item['label_path']} 错误信息:{item['error_message']}")

        if ENABLE_DELETE_INVALID:
            if sys.stdin.isatty():
                print("\n" + '=' * 60)
                print(f"检测到不合法数据集,是否删除这些不合法文件？")
                print("注意:删除操作将无法恢复,请谨慎操作!")
                print("1 是,删除图像与对应的标签文件")
                print("2 否,保留图像与标签文件")
                print("\n" + '=' * 60)

                user_choice = input("请输入你的选择 (1或者2) 1表示删除,2表示保留: ")
                if user_choice == '1':
                    delete_invalid_files(invalid_data_list)
                    basic_validation_problems_handled = True
                    logger.info("已删除不合法数据")
                elif user_choice == '2':
                    basic_validation_problems_handled = False
                    logger.info("已保留不合法数据")
                else:
                    logger.error("输入错误,不执行删除操作,不合法文件将会保留")
                    basic_validation_problems_handled = False
            else:
                logger.warning("当前环境非交互式终端,但是启动了删除功能,不合法文件将会删除")
                basic_validation_problems_handled = True
        else:
            logger.warning("当前环境非交互式终端,但未启动删除功能,不合法文件将会保留")
            basic_validation_problems_handled = False
    else:
        logger.info("基础数据集验证通过".center(60, "="))

    logger.info(f"开始数据集分割唯一性验证".center(60, "="))
    uniqueness_validation_passed = verify_split_uniqueness(CONFIGS_DIR / "data.yaml")
    if uniqueness_validation_passed:
       logger.info("数据集分割唯一性验证通过".center(60, "="))
    else:
        logger.error("数据集分割唯一性验证失败,存在重复图像或者标签,请查看日志")

    if basic_validation_problems_handled and uniqueness_validation_passed:
        logger.info("数据集验证通过".center(60, "="))
    else:
        logger.error("数据集验证未通过,请检查日志".center(60, "="))