"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - 全局配置模块 (防长度爆炸与显存优化版)
================================================================================
本模块控制 SFT / DPO / PPO / Eval 全流程的超参数与硬件调度。
包含针对长度爆炸 (Length-Bloat) 的 3 重防护约束。
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # 基础模型与数据路径配置
    model_name_or_path: str = field(
        default="../Qwen/Qwen3-1___7B",
        metadata={"help": "预训练基础模型本地路径或 HuggingFace ID"}
    )
    data_path: str = field(
        default="../qwen3_security_training_data.json",
        metadata={"help": "训练数据集本地 JSON/JSONL 路径"}
    )
    output_dir: str = field(
        default="./output_pipeline",
        metadata={"help": "检查点模型与评估报告输出根目录"}
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "HuggingFace Datasets 缓存目录"}
    )

    # 样本数量抽取控制配置 (提取到 Config 方便灵活切换小样本验证与全量训练)
    max_train_samples: Optional[int] = field(
        default=5000,  # ◄── 恢复 5000 篇全量规模高质量样本
        metadata={"help": "训练阶段最大抽样样本数 (None 表示全量数据)"}
    )
    max_eval_samples: int = field(
        default=300,  # ◄── 恢复 300 条评测样本量，消除误差假象
        metadata={"help": "评估阶段测试样本数量"}
    )

    # 文本长度配置 (防长度爆炸：设定 72 Tokens 极致精干客服区间)
    max_length: int = field(
        default=512,
        metadata={"help": "Token 序列最大截断长度"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Prompt 提示词最大截断长度"}
    )
    max_new_tokens: int = field(
        default=72,  # ◄── 防长度爆炸：硬限 72 Tokens (约 100~180 汉字黄金客服干货区间)
        metadata={"help": "推理与采样最大生成 Token 数量 (精简篇幅防唠叨)"}
    )

    # 硬件与显存安全配置 (RTX 4090 24GB 原生无损 BF16 优化)
    load_in_4bit: bool = field(
        default=False,  # ◄── 4090 24G 显存充裕，关闭 4-bit 量化压损，开启原生无损精度
        metadata={"help": "使用 BitsAndBytes NF4 4-bit 动态量化"}
    )
    use_bf16: bool = field(
        default=True,
        metadata={"help": "在支持的 GPU 上开启 bfloat16 混合精度"}
    )
    use_paged_optimizer: bool = field(
        default=False,  # ◄── 4090 显存充裕，直接使用标准 AdamW
        metadata={"help": "使用 Paged AdamW 8bit 显存分页优化器防止 OOM"}
    )

    # Batch Size 与梯度累积控制 (4090 吞吐优化)
    sft_batch_size: int = field(default=8)
    sft_gradient_accumulation_steps: int = field(default=2)
    dpo_batch_size: int = field(default=4)
    dpo_gradient_accumulation_steps: int = field(default=4)
    rm_batch_size: int = field(default=4)
    ppo_batch_size: int = field(default=4)
    ppo_mini_batch_size: int = field(default=2)

    # 训练 Epochs 与 Step 配置 (充分拟合 4 Epochs)
    sft_epochs: int = field(default=4)
    dpo_epochs: int = field(default=4)
    rm_epochs: int = field(default=3)
    ppo_steps: int = field(default=100)

    # 学习率配置
    sft_learning_rate: float = field(default=2e-4)
    dpo_learning_rate: float = field(default=5e-6)
    rm_learning_rate: float = field(default=1e-5)
    ppo_learning_rate: float = field(default=1.41e-5)

    # PEFT LoRA 配置
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # DPO 超参数 (防长度爆炸：提高 dpo_beta 到 0.30 加强 KL 散度约束)
    dpo_beta: float = field(
        default=0.30,  # ◄── 强化对偏离 Reference 产生冗长发散的惩罚
        metadata={"help": "DPO 隐式 Reward 比例系数 (越高对偏离惩罚越重)"}
    )
    dpo_loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO 损失函数类型 (sigmoid / cauchy_smooth)"}
    )
    num_iterative_rounds: int = field(
        default=2,
        metadata={"help": "Self-Play DPO 自进化迭代轮数"}
    )

    # 17 位数学巨匠算法开关配置
    use_sample_packing: bool = field(default=True, metadata={"help": "SFT 阶段 Sample Packing 样本拼合"})
    use_bidirectional_feedback: bool = field(default=True, metadata={"help": "DPO 阶段双向负反馈 (BNF)"})
    use_cauchy_smoothness: bool = field(default=True, metadata={"help": "柯西 C1 全纯 Softplus 与 5.0*tanh 截断"})
    use_kkt_dual_ascent: bool = field(default=True, metadata={"help": "拉格朗日 KKT 动态 Beta 对偶更新"})
    use_quantile_iqr_norm: bool = field(default=True, metadata={"help": "高斯 IQR 稳健归一化"})
    use_riemann_geodesic: bool = field(default=True, metadata={"help": "黎曼测地线正交范数保护惩罚"})
    use_position_swap_check: bool = field(default=True, metadata={"help": "伽罗瓦 S2 对称群双向位置校验"})
    use_llm_judge: bool = field(default=True, metadata={"help": "开启 LLM-as-a-Judge AlpacaEval 2.0 控长胜率评估"})

    # PPO / GRPO 强化学习超参数 (依据 SOTA 文档规范)
    use_grpo: bool = field(default=True, metadata={"help": "开启 GRPO 组相对优势算法 (废除 Critic 网络)"})
    grpo_group_size: int = field(default=4, metadata={"help": "GRPO 同一 Prompt 组采样数量 G=4"})
    non_eos_penalty: float = field(default=2.0, metadata={"help": "未吐出 EOS 结束符的强性惩罚值"})
    ppo_init_kl_coef: float = field(default=0.4)  # 加强 KL 散度惩罚至 0.4，强化贴近短回答分布

    def __post_init__(self):
        """校验配置参数合法性"""
        if self.load_in_4bit and not torch.cuda.is_available():
            print("⚠️ 警告: 未检测到 GPU CUDA 环境，load_in_4bit 将被自动禁用！")
            self.load_in_4bit = False

        os.makedirs(self.output_dir, exist_ok=True)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
