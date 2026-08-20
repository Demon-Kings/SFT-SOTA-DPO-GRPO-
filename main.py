"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - PyCharm 一键运行主程序入口 (显存安全版)
================================================================================
在 PyCharm 中右键点击本文件 main.py 并选择 "Run 'main'"，
系统将自动一键全量跑通：
1. SFT 监督微调 (Sample Packing + 4-bit/BF16 显存安全保护)
2. SOTA DPO 偏好对齐 (拒绝采样 + 长度归一化 + FocalPO + BNF Margin)
3. PPO 强化学习对齐 (RM 奖励模型 + Actor-Critic PPO + Reward 归一化)
4. DPO 与 PPO 模型多维自动化对比评测！
"""

import os
# 设置 PyTorch 显存碎片防护环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import sys
import json
import logging

# 配置标准日志打印输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from src.config import PipelineConfig
from src.sft_module import run_sft_pipeline
from src.dpo_module import run_dpo_pipeline
from src.rlhf_module import run_rm_pipeline, run_ppo_pipeline
from src.evaluator import ModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="大模型统一后训练与偏好对齐管线主入口")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",  # PyCharm 默认右键一键双跑 DPO + PPO
        choices=["sft", "dpo", "rm", "ppo", "eval", "all"],
        help="运行模式: sft (微调), dpo (对齐), rm (奖励模型), ppo (强化学习), eval (评测), all (一键全流程双跑 DPO+PPO)"
    )
    parser.add_argument("--model_path", type=str, default=None, help="覆盖模型路径")
    parser.add_argument("--data_path", type=str, default=None, help="覆盖数据路径")
    parser.add_argument("--output_dir", type=str, default="./output_pipeline", help="输出根目录")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.data_path:
        config.data_path = args.data_path
    if args.model_path:
        config.model_name_or_path = args.model_path

    logger.info(f"==================================================")
    logger.info(f"🚀 大模型后训练管线启动 | 当前模式: [{args.mode.upper()}]")
    logger.info(f"📍 模型路径: {config.model_name_or_path}")
    logger.info(f"📍 数据路径: {config.data_path}")
    logger.info(f"📍 显存模式: {'BitsAndBytes 4-bit 安全量化' if config.load_in_4bit else '无损 BF16 混合精度'}")
    logger.info(f"==================================================")

    if args.mode == "sft":
        run_sft_pipeline(config)

    elif args.mode == "dpo":
        run_dpo_pipeline(config)

    elif args.mode == "rm":
        run_rm_pipeline(config)

    elif args.mode == "ppo":
        rm_dir = run_rm_pipeline(config)
        run_ppo_pipeline(config, rm_model_dir=rm_dir)

    elif args.mode == "eval":
        eval_path = args.model_path if args.model_path else os.path.join(config.output_dir, "dpo_model")
        if not os.path.exists(eval_path):
            eval_path = config.model_name_or_path
        evaluator = ModelEvaluator(eval_path, config)
        evaluator.evaluate_dataset(num_samples=50)

    elif args.mode == "all":
        logger.info("🌟 启动一键全流程【DPO + PPO 双算法对齐与对比评测】模式...")
        
        # 1. 运行 SFT 监督微调
        logger.info("\n--------------------------------------------------")
        logger.info("第一阶段: 运行 SFT 监督微调...")
        sft_dir = run_sft_pipeline(config)
        
        # 2. 运行 DPO 偏好对齐
        logger.info("\n--------------------------------------------------")
        logger.info("第二阶段: 运行 SOTA DPO 偏好对齐...")
        dpo_dir = run_dpo_pipeline(config, sft_model_dir=sft_dir)
        
        # 3. 运行 PPO 强化学习对齐
        logger.info("\n--------------------------------------------------")
        logger.info("第三阶段: 运行 PPO 强化学习对齐 (RM 训练 + PPO 策略更新)...")
        rm_dir = run_rm_pipeline(config)
        ppo_dir = run_ppo_pipeline(config, rm_model_dir=rm_dir)
        
        # 4. 对 DPO 模型与 PPO 模型分别跑多维评估
        logger.info("\n--------------------------------------------------")
        logger.info("第四阶段: 运行 DPO 与 PPO 双模型多维对比自动化评估...")
        
        evaluator_dpo = ModelEvaluator(dpo_dir, config)
        report_dpo = evaluator_dpo.evaluate_dataset(num_samples=config.max_eval_samples)
        
        evaluator_ppo = ModelEvaluator(ppo_dir, config)
        report_ppo = evaluator_ppo.evaluate_dataset(num_samples=config.max_eval_samples)

        # 5. 导出双模型对比汇总报告
        summary_report = {
            "dpo_model": {
                "path": dpo_dir,
                "metrics": report_dpo["metrics"]
            },
            "ppo_model": {
                "path": ppo_dir,
                "metrics": report_ppo["metrics"]
            }
        }

        output_summary_file = os.path.join(config.output_dir, "eval", "dpo_vs_ppo_comparison_report.json")
        with open(output_summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)

        logger.info("\n🎉🎉🎉 DPO + PPO 全管线一键双跑成功完成！")
        logger.info(f"📁 DPO 模型路径: {dpo_dir}")
        logger.info(f"📁 PPO 模型路径: {ppo_dir}")
        logger.info(f"📊 DPO 评估指标: {report_dpo['metrics']}")
        logger.info(f"📊 PPO 评估指标: {report_ppo['metrics']}")
        logger.info(f"📄 双模型对比报告已保存至: {output_summary_file}")


if __name__ == "__main__":
    main()
