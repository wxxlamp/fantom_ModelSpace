# 修正后的完整类实现
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from typing import List
from .huggingface import HuggingFaceAgent

class LightweightHFAgent(HuggingFaceAgent):
    def __init__(self, model_name: str = "facebook/bart-base", **kwargs):
        super().__init__(**kwargs)

        # 新增配置参数
        self.device_map = kwargs.get("device_map", "auto")
        self.model_parallel = kwargs.get("model_parallel", True)

        # 初始化组件（移除旧的设备设置）
        self._load_model()
        self.init_pipeline()

    def _load_model(self):
        """修正后的模型加载方法"""
        from transformers import AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            truncation_side="left"
        )

        # 使用accelerate的自动设备映射
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map if self.model_parallel else None,
            low_cpu_mem_usage=True
        )

        # 非并行模式时手动移动设备
        if not self.model_parallel and torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def init_pipeline(self):
        """修正后的管道初始化"""
        # 移除所有device参数
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            torch_dtype=torch.float16,
            batch_size=self.batch_size,
            # 添加以下参数避免冲突
            device_map=self.device_map if self.model_parallel else None,
            enable_attention_masks=False  # 提升生成速度
        )

    def batch_interact(self, prompts: List[str], **kwargs) -> List[str]:
        """优化后的批量生成"""
        # 添加自动内存管理
        with torch.cuda.amp.autocast():
            outputs = self.pipe(
                [self.preprocess_input(p) for p in prompts],
                max_length=kwargs.get("max_length", 1024),
                num_beams=4,
                early_stopping=True,
                batch_size=self.batch_size,
                truncation_strategy="longest_first"  # 处理长文本
            )
        return [o['generated_text'].strip() for o in outputs]

    def generate(self, prompt: str, **kwargs) -> str:
        return self.batch_interact([prompt], **kwargs)[0]
