"""
================================================================================
大模型统一后训练 (Post-Training) 管线 - SFT 监督微调模块 (5,000 样本规模版)
================================================================================
数据与模型双重优化：
1. 扩展训练规模至 5,000 条精选问答。
2. Sample Packing 样本拼合零 Padding 计算浪费。
3. 动态 Tokenizer Padding-Side 与显存分页 AdamW 优化器。
"""

import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from src.config import PipelineConfig
from src.dataset import preprocess_sft_dataset

logger = logging.getLogger(__name__)


def run_sft_pipeline(config: PipelineConfig) -> str:
    """运行 SFT 监督微调主管道 (5,000 样本大规模版)"""
    logger.info(f"🔥 启动 SFT 大规模微调流程 (Sample Packing={config.use_sample_packing}, Paged Optimizer={config.use_paged_optimizer})...")

    output_dir = os.path.join(config.output_dir, "sft_model")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 载入 Tokenizer
    logger.info(f"📖 正在加载分词器: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 载入 Base Model
    compute_dtype = torch.bfloat16 if (config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = None
    if config.load_in_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    logger.info(f"🤖 正在加载大语言模型: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 挂载 PEFT / LoRA 适配器
    if config.use_lora:
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 4. 加载并拓展处理 5,000 条大规模数据
    logger.info(f"📥 正在加载主训练数据: {config.data_path}")
    if os.path.exists(config.data_path):
        raw_dataset = load_dataset('json', data_files=config.data_path, split='train', cache_dir=config.cache_dir)
        if config.max_train_samples is not None:
            max_s = min(config.max_train_samples, len(raw_dataset))
            logger.info(f"✂️ 根据 Config.max_train_samples 截取前 {max_s} 条样本进行 SFT 训练...")
            raw_dataset = raw_dataset.select(range(max_s))
        else:
            logger.info(f"🚀 未设置 max_train_samples 截断，使用全量数据 ({len(raw_dataset)} 条) 进行 SFT 训练！")
    else:
        logger.warning(f"⚠️ 找不到指定的训练数据文件: {config.data_path}，使用测试数据...")
        mock_data = [{"question": f"测试问题 {i}", "answer": f"测试正确回答 {i}"} for i in range(100)]
        raw_dataset = Dataset.from_list(mock_data)

    dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)

    logger.info("🧹 正在进行 SFT 数据清洗与 ChatML 编码...")
    processed_train = preprocess_sft_dataset(dataset_dict["train"], tokenizer, config)
    processed_val = preprocess_sft_dataset(dataset_dict["test"], tokenizer, config)

    optim_name = "paged_adamw_8bit" if (config.use_paged_optimizer and config.load_in_4bit) else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_gradient_accumulation_steps,
        learning_rate=config.sft_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=(not config.use_bf16 and torch.cuda.is_available()),
        bf16=(config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        optim=optim_name,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        data_collator=data_collator,
    )

    logger.info("🚀 启动 HuggingFace Trainer SFT 大规模微调...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ SFT 模型已保存至: {output_dir}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir
