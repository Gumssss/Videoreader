from transformers import AutoModelForCausalLM, AutoTokenizer

from base import BaseModule


# 加载模型和分词器
class StoryTellerV3(BaseModule):
    """StoryTeller
    =======================
    A module that generates a coherent story based on provided plot segments.
    It inherits from BaseModule.
    """
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

    def chat_Qwen3(self, output_language:str, system_prompt: str = "you are a helpful asistance.", max_tokens: int = 8192) -> str:
        
        if output_language == "中文":

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
            
            ## 第0步:判断输入语言
            如剧情片段的输入是英文，仍然输出中文

            ## 第一步：片段分析
            对每一条剧情片段，完成以下分析（不输出，只用于内部推理）：
            - 关键实体（人物 / 物品 / 地点）
            - 显性特征（直接描述的信息）
            - 隐性线索（可以推断出的时间、空间、情感或因果关系）

            ## 第二步：关系建模
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
        
        if output_language == "English":

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":f"""

            ## Plot Segments
            {self.text}

            ---

            ## Your Role
            You are a novelist skilled at constructing coherent narratives, adept at using **restrained and powerful language** to shape plots and characters. Please create a complete story based on the above plot segments.

            ---

            ## Writing Process (Please strictly follow these steps)

            ### Step 1: Segment Analysis
            For each plot segment, complete the following analysis (do not output, only for internal reasoning):
            - Key Entities (Characters / Items / Locations)
            - Explicit Features (Directly described information)
            - Implicit Clues (Time, space, emotional or causal relationships that can be inferred)

            ### Step 2: Relationship Modeling
            Based on the content of all segments, infer at least one coherent relationship between segments, including but not limited to:
            - Temporal Order (When events happen)
            - Spatial Migration (Transfer or change between different locations)
            - Causal Relationships (One segment leads to or triggers another)
            - Emotional Evolution (Development of character emotions)

            However, do not over-interpret or add non-existent plot segments.

            These relationships will help you naturally connect the segments when generating the story.

            ## Step 3: Story Generation (Must include the following requirements)

            1. Plot Importance Analysis (Internal processing, do not output)

            Please judge the importance of each user-provided plot segment (High / Medium / Low):
            - High: Decisive for main plot development or character transformation (Turning points, conflicts, decisions)
            - Medium: Supports the main plot but can be summarized (Transitions, background, clues)
            - Low: Provides supplementary information that can be quickly passed over or briefly mentioned (Secondary characters, locations, routine actions)

            When generating the story:
            - **High Importance Segments:** Expand details (actions/sensory/psychological/environmental)
            - **Medium Importance Segments:** Concise description + Simple transitions
            - **Low Importance Segments:** 1-2 sentences to mention, or even merged into other segments.

                The generated result should focus on describing high importance segments in detail, summarizing medium importance segments, and briefly mentioning low importance segments.


            2. **Natural Transitions**: Insert naturally connecting transition sentences between any two segments. You can start with phrases like:
            Smooth Transitions:
            “Meanwhile, …”

            “While … was happening, …”

            “Right after …, …”

            “Because …, … happened”
            3. **Rich in Detail**: Expand each segment by adding at least two specific details, such as:
            - Sensory descriptions (visual, auditory, tactile, olfactory, taste)
            - Environmental context (e.g., weather, lighting, landscape)
            - Action details (precise, vivid descriptions of the character's behavior)  
            4. Controlled Language Style:
            Avoid overusing adjectives, adverbs, or metaphors, especially the following types:
            Emotionally saturated phrases (e.g., “a night of unbearable sorrow”)
            Redundant modifiers (e.g., “sparkling silver-white snowflakes”)
            Limit to one adjective or adverb per sentence, unless absolutely necessary.
            Avoid excessive sensory or vague poetic language.   
            The writing style should be:
            - Concise and direct
            - Aligned with the narrative pace
            - Focused on showing emotions through actions and events, rather than abstract language
            5. **Output Format**: Directly return the final complete story text, without including analysis processes or adding explanations or comments.   
            """},
                    ]  
        # 生成 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,            
            add_generation_prompt=True, 
            enable_thinking=self.enable_thinking 
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Get the length of input tokens to extract only new generation
        input_length = inputs.input_ids.shape[1]

        # 生成输出
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=self.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract only the newly generated tokens (skip the input)
        generated_tokens = outputs[0][input_length:]

        # Decode only the generated part
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()

    def run(self, text: str, output_language: str = "English", system_prompt: str = "你是一个有帮助的助手。", max_tokens: int = 8192) -> str:
        """
        运行故事生成器，返回生成的故事文本。
        
        参数:
        - text (str): 输入的剧情片段文本。
        - output_language (str): 输出语言，默认为中文。 中文 or English
        - system_prompt (str): 系统提示语，默认为 "你是一个有帮助的助手。"。
        - max_tokens (int): 最大生成令牌数，默认为8192。
        
        返回:
        - str: 生成的故事文本。
        """
        self.get_text(text)
        return self.chat_Qwen3(output_language, system_prompt, max_tokens)
