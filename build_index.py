# build_index.py
import os
import json
import pandas as pd
import chromadb
from modelscope import snapshot_download
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

def main():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    csv_file = 'aee30b75-4d2a-41ef-bd7b-bc1cb1a30371.csv'
    
    print("1. 正在读取 PsyDTCorpus 心理咨询语料库...")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"找不到数据集文件 {csv_file}，请确保路径正确！")
        
    df_psy = pd.read_csv(csv_file)
    documents = []
    
    print("2. 执行【医患对话底层解耦】...")
    for idx, row in df_psy.iterrows():
        msgs = json.loads(row['messages'])
        tag = row.get('normalizedTag', '未知类别')
        
        system_theory, patient_utterances, doctor_responses = "", [], []
        
        for m in msgs:
            if m['role'] == 'system':
                system_theory = m['content']
            elif m['role'] == 'user':
                patient_utterances.append(m['content'])
            elif m['role'] == 'assistant':
                doctor_responses.append(m['content'])
                
        patient_text = " ".join(patient_utterances) 
        doctor_text = " ".join(doctor_responses)    
        
        doc = Document(
            text=patient_text, 
            metadata={
                "category": tag,
                "theory": system_theory[:500] if system_theory else "无", 
                "doctor_responses": doctor_text[:1000] if doctor_text else "无"
            },
            excluded_embed_metadata_keys=["category", "theory", "doctor_responses"],
            excluded_llm_metadata_keys=["category", "theory", "doctor_responses"]
        )
        documents.append(doc)

    print("3. 执行滑动窗口文本分块...")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    print("4. 拉取 GTE-zh 向量大模型...")
    model_dir = snapshot_download('iic/nlp_gte_sentence-embedding_chinese-base')
    embed_model = HuggingFaceEmbedding(model_name=model_dir)

    print("5. 初始化并写入本地 ChromaDB 索引库...")
    db_psy = chromadb.PersistentClient(path="./PsyDTCorpus_index")
    chroma_collection_psy = db_psy.get_or_create_collection("counseling_cases")
    vector_store_psy = ChromaVectorStore(chroma_collection=chroma_collection_psy)
    storage_context_psy = StorageContext.from_defaults(vector_store=vector_store_psy)

    VectorStoreIndex(
        nodes, 
        storage_context=storage_context_psy, 
        embed_model=embed_model,
        show_progress=True 
    )
    print("建库完成")

if __name__ == "__main__":
    main()
