"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - 数据工程与自博弈偏好生成模块 (防长度爆炸版)
================================================================================
本模块实现：
1. SFT 阶段：ChatML 协议封装、`-100` Label 损失掩码与 Sample Packing 样本拼合。
2. 拒绝采样 (Rejection Sampling) 偏好生成：
   - 8-Batch Left-Padding CUDA 矩阵化极速生成。
   - 【防长度爆炸约束】：提升 `len_ratio < 0.50` 严格控制好坏回答长度比例接近，防止筛选出唠叨废话。
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedModel
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def build_chat_prompt(question: str) -> str:
    """按阿里 Qwen 官方 ChatML 协议构造格式化 Prompt"""
    return (
        "<|im_start|>system\n"
        "你是智能客服专家，请简明扼要、条理清晰地回答用户的问题。<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def preprocess_sft_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    config: PipelineConfig
) -> Dataset:
    """SFT 阶段数据预处理 (带 ChatML 协议封装与 -100 Label 损失掩码)"""
    logger.info("🧹 开始预处理 SFT 监督微调数据集...")

    def encode_chat(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        full_prompt = build_chat_prompt(question)
        full_text = full_prompt + answer + "<|im_end|>"

        prompt_encoded = tokenizer(full_prompt, add_special_tokens=False)
        full_encoded = tokenizer(full_text, add_special_tokens=False, max_length=config.max_length, truncation=True)

        input_ids = full_encoded["input_ids"]
        attention_mask = full_encoded["attention_mask"]

        # -100 Label 掩码：将 Prompt 区域全部设为 -100
        prompt_len = len(prompt_encoded["input_ids"])
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    encoded_dataset = dataset.map(
        encode_chat,
        remove_columns=dataset.column_names,
        desc="执行 ChatML 格式化与 -100 掩码"
    )

    if config.use_sample_packing:
        logger.info("🧩 开启 Sample Packing (样本拼合)，拼接序列消灭 Padding 算力浪费...")

        def pack_examples(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            block_size = config.max_length

            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            return result

        encoded_dataset = encoded_dataset.map(
            pack_examples,
            batched=True,
            desc="执行 Sample Packing 序列拼合"
        )

    logger.info(f"✅ SFT 数据预处理完成，有效样本条数: {len(encoded_dataset)}")
    return encoded_dataset


def generate_self_play_preference_dataset(
    reference_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    config: PipelineConfig,
    num_samples: Optional[int] = None
) -> Dataset:
    """自博弈拒绝采样：左侧填充 8-Batch 极速生成 chosen vs rejected 偏好对 (防长度爆炸版)"""
    target_samples = num_samples if num_samples is not None else config.max_train_samples
    if target_samples is not None:
        logger.info(f"⚡ 启动 8-Batch Left-Padding 自博弈偏好生成器 (目标采样: {target_samples} 组)...")
    else:
        logger.info(f"⚡ 启动 8-Batch Left-Padding 自博弈偏好生成器 (使用全量数据集采样)...")

    # 设为 left-padding 防止生成警告与卡顿
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if os.path.exists(config.data_path):
        raw_ds = load_dataset('json', data_files=config.data_path, split='train', cache_dir=config.cache_dir)
        if target_samples is not None:
            raw_ds = raw_ds.select(range(min(target_samples * 2, len(raw_ds))))
    else:
        num_mock = (target_samples * 2) if target_samples is not None else 100
        raw_ds = Dataset.from_dict({
            "question": [f"退换货政策咨询 {i}" for i in range(num_mock)],
            "answer": [f"标准退换货解答流程说明 {i}" for i in range(num_mock)]
        })

    dpo_data = {"prompt": [], "chosen": [], "rejected": []}

    device = next(reference_model.parameters()).device
    reference_model.eval()

    batch_size = 8
    questions = [ex["question"] for ex in raw_ds]
    answers = [ex["answer"] for ex in raw_ds]

    with torch.no_grad():
        for b_idx in range(0, len(questions), batch_size):
            batch_q = questions[b_idx: b_idx + batch_size]
            batch_a = answers[b_idx: b_idx + batch_size]

            prompts = [build_chat_prompt(q) for q in batch_q]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length
            ).to(device)

            # 控制最大生成 Token 数并使用 temperature=0.5 降低发散冗余
            outputs = reference_model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_texts = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            for prompt, ref_ans, gen_ans in zip(prompts, batch_a, generated_texts):
                gen_len = len(gen_ans)
                ref_len = len(ref_ans)

                # 【防长度爆炸关键改动】：收紧限制 len_ratio < 0.35，严格控制好坏回答长度接近，防止挑出过长唠叨回答
                len_ratio = abs(gen_len - ref_len) / max(gen_len, ref_len, 1)

                if len(gen_ans.strip()) > 5 and len_ratio < 0.35:
                    dpo_data["prompt"].append(prompt)
                    dpo_data["chosen"].append(ref_ans)
                    dpo_data["rejected"].append(gen_ans)

    tokenizer.padding_side = orig_padding_side

    # 鲁棒保底机制
    if len(dpo_data["prompt"]) < 5:
        logger.warning("⚠️ 极严格长度过滤后偏好样本不足 5 组，注入鲁棒降级保底候选对...")
        for q, a in zip(questions[:10], answers[:10]):
            dpo_data["prompt"].append(build_chat_prompt(q))
            dpo_data["chosen"].append(a)
            dpo_data["rejected"].append(a + " (注: 此回复为系统自动生成的冗余说明)")

    pref_dataset = Dataset.from_dict(dpo_data)
    logger.info(f"✅ 自博弈拒绝采样偏好数据集构建完毕，成功收集 {len(pref_dataset)} 对精干偏好数据！")
    return pref_dataset
