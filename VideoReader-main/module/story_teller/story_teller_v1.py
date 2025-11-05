from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 加载模型和分词器
class StoryTeller:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="bfloat16"
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            trust_remote_code=True
        )
        self.enable_thinking=False
        self.temperature=0.7
        self.top_p=0.8
        self.top_k=20
    
    def get_text(self, text: str) -> str:
        self.text = text   

    def chat_Qwen3(self, system_prompt: str = "you are a helpful asistance.", max_tokens: int = 8192) -> str:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":f"""

## 剧情片段
{self.text}

---

## 你的角色
你是一位擅长构建连贯叙事的小说作家，擅长使用**克制、有力的语言**来塑造情节与人物,请基于以上剧情片段创作一个完整故事。

---

## 写作流程（请严格遵循以下步骤）

### 第一步：片段分析
对每一条剧情片段，完成以下分析（不输出，只用于内部推理）：
- 关键实体（人物 / 物品 / 地点）
- 显性特征（直接描述的信息）
- 隐性线索（可以推断出的时间、空间、情感或因果关系）

### 第二步：关系建模
根据所有片段的内容，推断片段之间至少一种连贯关系，包括但不限于：
- 时间顺序（先后发生）
- 空间迁移（不同地点之间的转移或变化）
- 因果关系（一个片段导致或引发另一个）
- 情绪演进（角色情感的发展）

但是不要过度联想，和添加不存在的剧情片段。
这些关系将帮助你在生成故事时自然衔接各个片段。

## 第三步：故事生成（必须包含以下要求）

1. 情节重要度分析（内部处理，不输出）

请判断每个用户提供的剧情片段的重要程度（高 / 中 / 低）：
- 高：对主线情节发展或人物转变起到决定作用（转折、冲突、决定）
- 中：支持主线，但可以简写（过渡、背景、线索）
- 低：仅提供补充信息，可快速带过或略写（次要人物、地点、常规行为）

你将在生成故事时：
- **高重要度片段：** 扩展细节（动作/感官/心理/环境）
- **中重要度片段：** 精炼描述 + 简洁衔接
- **低重要度片段：** 1-2句交代，甚至合并进其他段。

    生成结果应着重描写高重要度片段，简写中重要度片段，略写低重要度片段。


2. **过渡自然**：在任意两个片段之间，插入自然衔接的过渡句。可以使用以下方式开头：
   - “与此同时，……”
   - “当……发生时，……”
   - “就在……之后，……”
   - “因为……，……发生了”

3. **细节丰富**：每个片段基础上至少扩展2处具体细节，例如：
   - 感官描写（视觉 / 听觉 / 触觉 / 嗅觉 / 味觉）
   - 环境刻画（天气 / 光线 / 地貌等）
   - 动作细节（角色行为的具体描写）

4. 语言风格控制：
    禁止滥用形容词、副词和比喻，尤其是以下类型：
    情绪过饱和（如“极度悲伤的黑夜”）
    多余修饰（如“闪烁着银光的白色雪花”）
    每句最多使用 **一个形容词或副词**，除非确有必要。
    避免堆砌感官词汇或空泛描写。
    语言风格应：
    凝练、直接
    贴合情节节奏
    让情感通过**行为和事件**体现，而非空洞修辞
5. **输出格式**：直接返回最终完整故事正文，不包含分析过程、不添加解释或注释。

        """},
        ]
        
        # 生成 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,            # 先得到纯字符串，方便调试查看
            add_generation_prompt=True, # 在末尾加 "<|assistant|>" 起始标记
            enable_thinking=self.enable_thinking # 是否启用思考模式
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 生成输出
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        
        # 解码输出并返回
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    
def main():
    run = StoryTeller()
    run.get_text(
        "一个男人走进了厨房。\n"
        "他打开了冰箱的门，从冰箱里取出了一盒鸡蛋。\n"
        "他把鸡蛋放     在料理台上，点燃了炉灶的火。\n"
        "他在平底锅里倒了一些油，油热后打了一个鸡蛋进去。\n"
        "鸡蛋在锅里发出滋滋的响声，他用铲子轻轻翻动鸡蛋。\n"
        "鸡蛋渐渐变成了金黄色的煎蛋，他关掉了炉火。\n"
        "从橱柜里拿出一个盘子，把煎蛋盛到了盘子里，又在煎蛋上撒了些盐和胡椒。\n"
        "他取出一片面包放入烤面包机，面包烤好后发出\"叮\"的一声。\n"
        "他把烤好的面包也放在盘子里，从冰箱拿出果汁倒了一杯。\n"
        "男人端着早餐走向餐桌，坐下来开始享用他的早餐。"
    )
    result = run.chat_Qwen3()
    print(result)

main()