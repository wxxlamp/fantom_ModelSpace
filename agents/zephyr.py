import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .huggingface import HuggingFaceAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.cuda.empty_cache()

class ZephyrAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 加载CPU优化模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",
            padding_side='left'
        )

        # CPU专用加载配置
        self.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.float32,  # CPU上使用float32更稳定
            low_cpu_mem_usage=True,
            **kwargs
        )

        # 应用动态量化（关键优化）
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        ).to(device)

        # 模板设置
        self.chat_template = {
            "system": "<|system|>{content}</s>",
            "user": "<|user|>{content}</s>",
            "assistant": "<|assistant|>{content}</s>"
        }

        # 内存优化配置
        self.model.config.use_cache = False
        self.model.eval()  # 固定为推理模式

        # 初始化CPU优化的pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # 强制使用CPU
        )

    def preprocess_input(self, text):
        """缩短输入长度"""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text[:512]}],  # 限制输入长度
            tokenize=False,
            add_generation_prompt=True
        )

    def postprocess_output(self, output):
        """优化内存的响应处理"""
        return output.split("<|assistant|>")[-1].strip().replace("</s>", "")[:500]  # 限制输出长度

    def generate(self, prompt, temperature=0.7, max_tokens=128, **kwargs):  # 大幅减少max_tokens
        formatted_prompt = self.preprocess_input(prompt)

        # CPU优化生成参数
        generation_args = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.3),  # 防止过低温度导致内存波动
            "top_p": 0.85,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,  # 仅生成单个序列
            "no_repeat_ngram_size": 3  # 减少内存占用
        }

        try:
            # 分块生成以控制内存
            outputs = []
            for chunk in self.pipe(
                    formatted_prompt,
                    **generation_args,
                    stream=True,  # 启用流式生成
                    chunk_size=32  # 小分块处理
            ):
                outputs.append(chunk[0]['generated_text'])
                if len(outputs) >= (max_tokens // 32):
                    break
            return self.postprocess_output(''.join(outputs))
        finally:
            # 手动清理内存
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
