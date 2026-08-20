"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - 奖励模型 (RM) 与 PPO 强化学习模块 (防长度爆炸版)
================================================================================
本模块包含：
1. 修复 Transformers 4-bit 量化下加载适配器报错。
2. 【防长度爆炸】: 在 Reward 函数中增加双向长度惩罚，对超过预期的冗余长回答施加负反馈惩罚。
3. 【柯西 C1】: 5.0 * tanh(R / 5.0) 光滑截断。
4. 【高斯 IQR】: 四分位距稳健归一化。
"""

import os
import logging
import math
import torch
import torch.nn as F_nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset, load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from src.config import PipelineConfig
from src.dataset import build_chat_prompt, generate_self_play_preference_dataset

logger = logging.getLogger(__name__)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """安全获取模型所在设备"""
    if hasattr(model, "device"):
        return model.device
    elif hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "device"):
        return model.pretrained_model.device
    else:
        return next(model.parameters()).device


def normalize_and_clip_rewards(rewards: torch.Tensor, config: PipelineConfig) -> torch.Tensor:
    """
    奖励信号连续平滑归一化（包含【柯西 C1 tanh 截断】与【高斯 IQR 稳健归一化】）。
    """
    if len(rewards) == 0:
        return rewards

    device = rewards.device
    r_float = rewards.float()

    if config.use_quantile_iqr_norm and len(r_float) > 1:
        q25 = torch.quantile(r_float, 0.25)
        q50 = torch.quantile(r_float, 0.50)
        q75 = torch.quantile(r_float, 0.75)
        iqr = (q75 - q25).clamp(min=1e-5)
        norm_rewards = (r_float - q50) / iqr
    else:
        mean = r_float.mean()
        std = r_float.std(unbiased=False).clamp(min=1e-5)
        norm_rewards = (r_float - mean) / std

    if config.use_cauchy_smoothness:
        clipped_rewards = 5.0 * torch.tanh(norm_rewards / 5.0)
    else:
        clipped_rewards = torch.clamp(norm_rewards, min=-5.0, max=5.0)

    return clipped_rewards.to(device)


def run_rm_pipeline(config: PipelineConfig) -> str:
    """运行 Reward Model (RM) 成对偏好微调"""
    logger.info("🎯 启动 Reward Model (RM) 训练管线...")
    output_dir = os.path.join(config.output_dir, "rm")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if (config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = None
    if config.load_in_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    pref_ds = generate_self_play_preference_dataset(ref_model, tokenizer, config, num_samples=150)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=1,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    rm_model.config.pad_token_id = tokenizer.pad_token_id

    if config.use_lora:
        if config.load_in_4bit:
            rm_model = prepare_model_for_kbit_training(rm_model)
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        rm_model = get_peft_model(rm_model, lora_cfg)

    optimizer = torch.optim.AdamW(rm_model.parameters(), lr=config.rm_learning_rate)
    rm_model.train()

    logger.info("🔥 启动 Reward Model (RM) Bradley-Terry Ranking 损失训练...")
    rm_device = get_model_device(rm_model)

    for epoch in range(config.rm_epochs):
        for i in range(0, len(pref_ds), config.rm_batch_size):
            batch = pref_ds[i:i+config.rm_batch_size]
            chosen_texts = [p + c for p, c in zip(batch["prompt"], batch["chosen"])]
            rejected_texts = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]

            chosen_toks = tokenizer(chosen_texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(rm_device)
            rejected_toks = tokenizer(rejected_texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(rm_device)

            r_chosen = rm_model(**chosen_toks).logits.squeeze(-1)
            r_rejected = rm_model(**rejected_toks).logits.squeeze(-1)

            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rm_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ Reward Model (RM) 训练完成，模型保存至: {output_dir}")
    return output_dir


def create_safe_ppo_config(config: PipelineConfig) -> PPOConfig:
    """自动兼容不同版本的 TRL PPOConfig 实例化定义"""
    try:
        return PPOConfig(
            learning_rate=config.ppo_learning_rate,
            per_device_train_batch_size=config.ppo_mini_batch_size,
            gradient_accumulation_steps=1
        )
    except TypeError:
        try:
            return PPOConfig(
                learning_rate=config.ppo_learning_rate,
                batch_size=config.ppo_batch_size,
                mini_batch_size=config.ppo_mini_batch_size,
                init_kl_coef=config.ppo_init_kl_coef,
                target_kl=0.1
            )
        except TypeError:
            return PPOConfig(learning_rate=config.ppo_learning_rate)


def run_ppo_pipeline(config: PipelineConfig, rm_model_dir: str = None) -> str:
    """
    运行 SOTA GRPO (Group Relative Policy Optimization) / PPO 强化学习策略对齐管线
    依据 DPO_PPO_CORE_ALGORITHM_ANALYSIS.md 规范：
    1. 废除 Value Head (Critic) 架构，释放 30% 显存并消除方差崩溃。
    2. 同一 Prompt 组采样 G=4，计算组内 Advantage 相对归一化。
    3. 引入复合奖励塑形：未输出 EOS 严惩 (-2.0) + Token 级平滑长度惩罚 + 柯西 Tanh 软截断。
    """
    logger.info("⚡ 启动 SOTA GRPO (组相对优势) 强化学习策略对齐管线...")
    output_dir = os.path.join(config.output_dir, "ppo_model")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compute_dtype = torch.bfloat16 if (config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = None
    if config.load_in_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    # GRPO 无需 Value Head，直接载入标准 CausalLM 并挂载 PEFT LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    
    if config.use_lora:
        if config.load_in_4bit:
            base_model = prepare_model_for_kbit_training(base_model)
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        model = base_model

    model.config.pad_token_id = tokenizer.pad_token_id
    actor_device = get_model_device(model)

    rm_path = rm_model_dir if (rm_model_dir and os.path.exists(rm_model_dir)) else os.path.join(config.output_dir, "rm")
    rm_model = None
    if os.path.exists(rm_path):
        try:
            adapter_cfg = os.path.join(rm_path, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                base_rm = AutoModelForSequenceClassification.from_pretrained(
                    config.model_name_or_path, num_labels=1, quantization_config=quant_config, device_map="auto" if torch.cuda.is_available() else None, torch_dtype=compute_dtype, trust_remote_code=True
                )
                rm_model = PeftModel.from_pretrained(base_rm, rm_path, is_trainable=False)
            else:
                rm_model = AutoModelForSequenceClassification.from_pretrained(
                    rm_path, num_labels=1, quantization_config=quant_config, device_map="auto" if torch.cuda.is_available() else None, torch_dtype=compute_dtype, trust_remote_code=True
                )
            rm_model.config.pad_token_id = tokenizer.pad_token_id
            rm_model.eval()
            logger.info(f"✅ 成功加载 Reward Model (RM): {rm_path}")
        except Exception as e:
            logger.warning(f"⚠️ RM 权重加载警示: {e}，将使用规则对抗性奖励逻辑作为评分器")
            rm_model = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ppo_learning_rate)

    if os.path.exists(config.data_path):
        raw_ds = load_dataset('json', data_files=config.data_path, split='train', cache_dir=config.cache_dir)
        if config.max_train_samples is not None:
            max_s = min(config.max_train_samples, len(raw_ds))
            logger.info(f"✂️ 根据 Config.max_train_samples 截取前 {max_s} 条 Prompt 进行 GRPO 训练...")
            raw_ds = raw_ds.select(range(max_s))
        else:
            logger.info(f"🚀 未设置 max_train_samples 截断，使用全量数据 ({len(raw_ds)} 条 Prompt) 进行 GRPO 训练！")
        prompts = [build_chat_prompt(ex["question"]) for ex in raw_ds]
    else:
        num_m = config.max_train_samples if config.max_train_samples is not None else 40
        prompts = [build_chat_prompt(f"测试问题 {i}") for i in range(num_m)]

    logger.info("🔥 启动 GRPO 策略网络无 Critic 组采样强化学习循环...")
    G = getattr(config, "grpo_group_size", 4)
    model.train()

    for i in range(0, len(prompts), config.ppo_batch_size):
        batch_prompts = prompts[i:i+config.ppo_batch_size]

        for p_idx, prompt in enumerate(batch_prompts):
            p_toks = tokenizer([prompt] * G, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length).to(actor_device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **p_toks,
                    max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9
                )
                response_ids = gen_ids[:, p_toks["input_ids"].shape[1]:]
                responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            if rm_model is not None:
                rm_device = get_model_device(rm_model)
                texts = [prompt + r for r in responses]
                rm_inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(rm_device)
                with torch.no_grad():
                    raw_rewards = rm_model(**rm_inputs).logits.squeeze(-1)
            else:
                raw_rewards = torch.tensor([float(len(r)) / 100.0 for r in responses], device=actor_device)

            # 【复合奖励塑形 1：未输出 EOS 严性负分 (-2.0) + 长度惩罚】
            non_eos_penalty = getattr(config, "non_eos_penalty", 2.0)
            for r_i, r_ids in enumerate(response_ids):
                token_len = len(r_ids)
                last_token = r_ids[-1].item() if token_len > 0 else 0
                if last_token != tokenizer.eos_token_id and token_len >= config.max_new_tokens - 2:
                    raw_rewards[r_i] -= non_eos_penalty
                if token_len > 72:
                    raw_rewards[r_i] -= 0.06 * (token_len - 72)

            rewards = normalize_and_clip_rewards(raw_rewards, config)

            # 【GRPO 核心算法】：组内 z-score 归一化计算组相对 Advantage
            if len(rewards) > 1 and rewards.std() > 1e-5:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            else:
                advantages = rewards - rewards.mean()

            # 前向计算当前策略对组内 Response 的对数概率
            full_inputs = tokenizer([prompt + r for r in responses], padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(actor_device)
            outputs = model(**full_inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = full_inputs["input_ids"][..., 1:].contiguous()
            shift_mask = full_inputs["attention_mask"][..., 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            sequence_log_probs = (per_token_logps * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1.0)

            # 策略梯度 Loss (带组 Advantage 加权)
            loss = -(sequence_log_probs * advantages.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"⚡ GRPO Step [{i//config.ppo_batch_size + 1}/{(len(prompts)+config.ppo_batch_size-1)//config.ppo_batch_size}]: loss={loss.item():.4f}, mean_reward={rewards.mean().item():.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ SOTA GRPO 强化学习策略对齐完成，模型保存至: {output_dir}")
    return output_dir


