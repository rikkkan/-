# qa_engine.py
import os
import torch
import chromadb
from modelscope import snapshot_download
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer

class PsyQAEngine:
    def __init__(self):
        print("初始化系统引擎中，请稍候...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 1. 加载本地向量库与检索器
        embed_model_dir = snapshot_download('iic/nlp_gte_sentence-embedding_chinese-base')
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_dir)
        
        self.db_psy = chromadb.PersistentClient(path="./PsyDTCorpus_index")
        self.chroma_collection = self.db_psy.get_collection("counseling_cases")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, embed_model=self.embed_model
        )
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        
        # 2. 加载 Qwen 1.5 语言模型 (FP16半精度)
        llm_dir = snapshot_download('qwen/Qwen1.5-4B-Chat')
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_dir, device_map="auto", torch_dtype=torch.float16
        )
        print("✅ 引擎初始化完成，Qwen 已就位！")

    def generate_reply(self, user_message, history):
        """处理单次对话逻辑"""
        # A. 检索历史专家话术
        retrieved_nodes = self.retriever.retrieve(user_message)
        reference_cases_str = ""
        for i, node in enumerate(retrieved_nodes):
            theory = node.metadata.get('theory', '通用支持')
            doctor_words = node.metadata.get('doctor_responses', '倾听')
            reference_cases_str += f"- 案例{i+1}: 理论[{theory}], 话术[{doctor_words}]\n"

        # B. 组装 System Prompt (设置护栏)
        system_prompt = f"""你是一位拥有20年临床经验的顶级心理咨询师。
你的任务是通过多轮对话引导来访者。
【系统为你检索到的历史相似病历干预话术参考】：
{reference_cases_str}
【最高指令约束】：
1. 绝对不要长篇大论！每次只回复 1 到 3 句话！
2. 不要急于给出建议！要顺着对方的话题进行共情。
3. 你的回复结尾，通常应该是一个温和的、启发式的反问句。"""

        # C. 历史记忆拼接
        messages = [{"role": "system", "content": system_prompt}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": user_message})

        # D. 模型推理
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=150, temperature=0.6, top_p=0.8
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
