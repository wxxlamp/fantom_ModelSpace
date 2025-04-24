from modelscope import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from .base import BaseAgent

class DeepSeekAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

        # 获取设备信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.max_memory = 15 * 1024**3  # 20GB安全阈值
        self.batch_size = 1  # 根据显存动态调整

        # 简化参数设置
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 10240)
        self.top_p = kwargs.get('top_p', 0.95)

        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            **kwargs
        )

        # 设置对话模板
        self.chat_template = [
            {"role": "system", "content": "You are a helpful assistant focusing on Situational understanding, give me the answer directly without reasoning"},
            {"role": "user", "content": "\n{content}\n"}
        ]

    def preprocess_input(self, text):
        """将输入文本转换为适合模型的格式"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant focusing on Situational understanding"},
                {"role": "user", "content": text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)

    def postprocess_output(self, output):
        pass

    def generate(self, prompt, temperature=None, max_tokens=None):
        raise NotImplementedError

    def _check_memory(self):
        """实时监控显存使用"""
        used = torch.cuda.memory_allocated()
        if used > self.max_memory:
            self.batch_size = max(1, self.batch_size // 2)
            torch.cuda.empty_cache()

    def interact(self, prompt):
        self._check_memory()
        pre = self.preprocess_input(prompt)

        generated_ids = self.model.generate(
            **pre,
            max_new_tokens=2048,
            temperature = 0.1,
            # 新增防循环参数
            repetition_penalty=1.2,  # 重复惩罚系数（1.0表示无惩罚）
            no_repeat_ngram_size=3,   # 禁止3-gram重复
            do_sample=True,          # 启用采样模式
            top_k=50,                # 限制采样候选词数量
            top_p=0.95               # 核采样概率阈值
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(pre.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self._check_memory()

        try:
            return response.split('</think>')[1].strip()
        except (AttributeError, IndexError):
            return response.strip()

    def batch_generate(self, prompts, temperature=None, max_tokens=None):
        """批量生成回复"""
        return [self.generate(prompt, temperature, max_tokens) for prompt in prompts]

    def batch_interact(self, texts):
       raise NotImplementedError

    def batch_interact_dep(self, texts):
        """优化批量推理"""
        self._check_memory()

        # 自动批处理
        responses = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=10240,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend([self.postprocess_output(r) for r in batch_responses])

            # 输出当前批次的响应到debug文件中（JSON格式）
            with open('debug', 'a') as f:
                import json
                debug_info = {"batch_responses": batch_responses, "responses": responses}
                f.write(json.dumps(outputs.tolist()) + "\n")  # 将Tensor转换为列表再写入文件
                f.write(json.dumps(debug_info) + "\n")

            self._check_memory()  # 每个批次后检查显存
            raise NotImplementedError
        return responses
