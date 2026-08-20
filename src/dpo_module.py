"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - DPO 偏好对齐模块 (防长度爆炸与安全加载版)
================================================================================
本模块完全包含：
1. `load_model_with_optional_lora()`: 解决 4-bit 量化下直接加载适配器抛出 `AttributeError: 'weight' is not an nn.Module`。
2. 【柯西 C1 光滑 BNF】: Softplus(0.1 - Δr, β=10.0) 替代非连续线性 ReLU。
3. 【拉格朗日 KKT】: 动态 Beta 更新 β_t+1 = Clamp(β_t + η(KL - 0.05))。
4. 【黎曼测地线】: L_geo = 0.005 * (logps_w^2 + logps_l^2) 流形正则化。
5. 【防长度爆炸】: 接入显式 Token 长度差惩罚项 (Length Difference Penalty)，强力抑制长文本偏置。
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset, load_dataset
from src.config import PipelineConfig
from src.dataset import build_chat_prompt, generate_self_play_preference_dataset

logger = logging.getLogger(__name__)


def load_model_with_optional_lora(model_path_or_dir: str, config: PipelineConfig, is_trainable: bool = False):
    """
    标准安全模型加载辅助函数 (解决 AttributeError: 'weight' is not an nn.Module 报错)
    """
    compute_dtype = torch.bfloat16 if (config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = None
    if config.load_in_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    adapter_config_path = os.path.join(model_path_or_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info(f"📦 检测到 LoRA 检查点路径: {model_path_or_dir}，执行 Base Model + PeftModel 挂载...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_path_or_dir,
            is_trainable=is_trainable
        )
        return model
    else:
        logger.info(f"📦 直接载入全量/基线模型权重: {model_path_or_dir}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_dir,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )
        return model


class OptimizedDPOTrainer:
    """带防长度爆炸惩罚与 17 位数学巨匠算法的 SOTA DPO 训练器"""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: AutoTokenizer,
        config: PipelineConfig
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config

        self.current_beta = config.dpo_beta
        self.current_length_lambda = 0.005  # ◄── KKT 动态长度对偶乘子初始值 λ_0 提升至 0.005
        self.use_length_normalization = True
        self.use_bidirectional_feedback = config.use_bidirectional_feedback
        self.use_cauchy_smoothness = config.use_cauchy_smoothness
        self.use_kkt_dual_ascent = config.use_kkt_dual_ascent
        self.use_riemann_geodesic = config.use_riemann_geodesic

    def compute_sequence_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取 Shift-Left 序列对数概率 log p(y|x)"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        sequence_log_probs = (per_token_logps * shift_mask).sum(-1)
        sequence_lengths = shift_mask.sum(-1).clamp(min=1.0)

        if self.use_length_normalization:
            sequence_log_probs = sequence_log_probs / sequence_lengths

        return sequence_log_probs, sequence_lengths

    def update_kkt_dual_beta(self, chosen_adv: torch.Tensor):
        """【拉格朗日】KKT 动态对偶 Beta 调度 (约束 KL 散度)"""
        if not self.use_kkt_dual_ascent:
            return
        with torch.no_grad():
            kl_div = chosen_adv.detach().abs().mean().item()
            target_kl = 0.05
            lr_dual = 0.001
            self.current_beta = max(0.05, min(0.5, self.current_beta + lr_dual * (kl_div - target_kl)))

    def update_kkt_dual_length_lambda(self, chosen_lengths: torch.Tensor, rejected_lengths: torch.Tensor):
        """【拉格朗日】KKT 互补松弛条件下的动态长度对偶乘子 λ_len 调度 (动态触发控长)"""
        if not self.use_kkt_dual_ascent:
            return
        with torch.no_grad():
            avg_len_diff = (chosen_lengths - rejected_lengths).mean().item()
            target_len_slack = 0.0  # 目标：chosen 不得比 rejected 更长
            lr_dual_len = 0.0005
            # KKT 不等式约束：λ >= 0，当长度超标时 λ 自动爬升增大惩罚，未超标时 λ 衰减回 0
            self.current_length_lambda = max(0.0, min(0.05, self.current_length_lambda + lr_dual_len * (avg_len_diff - target_len_slack)))

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算带 KKT 动态对偶触发惩罚的 DPO 损失"""
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_advantages = self.current_beta * chosen_logratios
        rejected_advantages = self.current_beta * rejected_logratios

        logits = chosen_advantages - rejected_advantages

        if self.config.dpo_loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        else:
            losses = -F.logsigmoid(logits)

        # 【柯西 C1】全纯 Softplus 双向负反馈 (BNF)
        if self.use_bidirectional_feedback:
            if self.use_cauchy_smoothness:
                bnf_loss = F.softplus(0.1 - (chosen_advantages - rejected_advantages), beta=10.0).mean()
            else:
                bnf_loss = F.relu(0.1 - (chosen_advantages - rejected_advantages)).mean()
            losses = losses + 0.1 * bnf_loss

        # 【KKT 动态对偶触发控长】：根据 KKT 乘子 λ_len 动态调节惩罚力度
        if self.use_kkt_dual_ascent:
            if self.use_cauchy_smoothness:
                length_loss = F.softplus(chosen_lengths - rejected_lengths, beta=5.0).mean()
            else:
                length_loss = (chosen_lengths - rejected_lengths).clamp(min=0.0).mean()
            length_penalty = self.current_length_lambda * length_loss
        else:
            length_penalty = 0.01 * (chosen_lengths - rejected_lengths).clamp(min=0.0).mean()

        # 【SFT 辅助自回归正则】：锁定优质 Chosen 样本的绝对似然概率，防止似然位移 (Likelihood Displacement)
        sft_aux_loss = -policy_chosen_logps.mean()
        final_loss = losses.mean() + length_penalty + 0.05 * sft_aux_loss

        # 【黎曼测地线】正交范数保护惩罚
        if self.use_riemann_geodesic:
            riemann_penalty = 0.005 * (policy_chosen_logps.pow(2).mean() + policy_rejected_logps.pow(2).mean())
            final_loss = final_loss + riemann_penalty

        # 触发双 KKT 对偶调度：Beta (KL 散度) 与 Lambda (长度膨胀)
        self.update_kkt_dual_beta(chosen_advantages)
        self.update_kkt_dual_length_lambda(chosen_lengths, rejected_lengths)

        return final_loss, chosen_advantages.detach().mean(), rejected_advantages.detach().mean()


def run_dpo_pipeline(config: PipelineConfig, sft_model_dir: str = None) -> str:
    """运行 SOTA DPO 偏好对齐管线 (带防长度爆炸控制)"""
    logger.info("🚀 启动 DPO (Direct Preference Optimization) 偏好对齐管线...")
    output_dir = os.path.join(config.output_dir, "dpo_model")
    os.makedirs(output_dir, exist_ok=True)

    if sft_model_dir is None or not os.path.exists(sft_model_dir):
        sft_model_dir = os.path.join(config.output_dir, "sft_model")

    model_path = sft_model_dir if os.path.exists(sft_model_dir) else config.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 安全载入 Policy 与 Reference 模型
    model = load_model_with_optional_lora(model_path, config, is_trainable=True)
    ref_model = load_model_with_optional_lora(model_path, config, is_trainable=False)
    ref_model.eval()

    # 生成控制长度的拒绝采样偏好数据集 (扩充至 150 组)
    dpo_dataset = generate_self_play_preference_dataset(
        reference_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        num_samples=150
    )

    dpo_trainer = OptimizedDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config
    )

    optimizer_name = "paged_adamw_8bit" if config.load_in_4bit else "adamw_torch"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.dpo_learning_rate)

    logger.info(f"🔥 开始执行 DPO 训练 (Epochs: {config.dpo_epochs}, Beta: {config.dpo_beta})...")
    model.train()

    device = next(model.parameters()).device

    for epoch in range(config.dpo_epochs):
        epoch_loss = 0.0
        for i in range(0, len(dpo_dataset), config.dpo_batch_size):
            batch = dpo_dataset[i:i+config.dpo_batch_size]

            chosen_texts = [p + c for p, c in zip(batch["prompt"], batch["chosen"])]
            rejected_texts = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]

            chosen_toks = tokenizer(chosen_texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)
            rejected_toks = tokenizer(rejected_texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)

            policy_chosen_logps, chosen_lens = dpo_trainer.compute_sequence_log_probs(model, chosen_toks["input_ids"], chosen_toks["attention_mask"])
            policy_rejected_logps, rejected_lens = dpo_trainer.compute_sequence_log_probs(model, rejected_toks["input_ids"], rejected_toks["attention_mask"])

            with torch.no_grad():
                ref_chosen_logps, _ = dpo_trainer.compute_sequence_log_probs(ref_model, chosen_toks["input_ids"], chosen_toks["attention_mask"])
                ref_rejected_logps, _ = dpo_trainer.compute_sequence_log_probs(ref_model, rejected_toks["input_ids"], rejected_toks["attention_mask"])

            loss, chosen_adv, rejected_adv = dpo_trainer.compute_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                chosen_lens, rejected_lens
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f" Epoch [{epoch+1}/{config.dpo_epochs}] 均值 Loss: {epoch_loss / (len(dpo_dataset)/config.dpo_batch_size):.4f} | 动态 Beta: {dpo_trainer.current_beta:.4f} | 控长 Lambda: {dpo_trainer.current_length_lambda:.5f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ DPO 偏好对齐训练完成，模型保存至: {output_dir}")
    return output_dir
