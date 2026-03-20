import datetime

class DualTrackMemory:
    def __init__(self, long_term_collection, window_size=3):
        """
        初始化双轨制记忆引擎
        :param long_term_collection: 已经实例化的 ChromaDB 集合句柄
        :param window_size: 短期记忆保留的最大对话轮数
        """
        self.window_size = window_size
        self.short_term_buffer = []       # 内存中的短期滑动窗口 (List)
        self.long_term_db = long_term_collection # 磁盘上的长期向量记忆库 (ChromaDB)

    def update_and_archive(self, user_msg, ai_msg):
        """
        动作 1：更新短期记忆，并将溢出的古老记忆异步落盘
        """
        # 1. 当前轮次入队
        self.short_term_buffer.append({"role": "user", "content": user_msg})
        self.short_term_buffer.append({"role": "assistant", "content": ai_msg})
        
        # 2. 检查是否溢出 (乘 2 因为一轮包含问答两句)
        if len(self.short_term_buffer) > self.window_size * 2:
            # 3. 触发滑动截断：弹出最老的一轮对话
            oldest_user = self.short_term_buffer.pop(0)
            oldest_ai = self.short_term_buffer.pop(0)
            
            # 4. 组装为长期记忆片段，打上 Metadata 标签，落盘持久化
            archive_text = f"患者曾表述: {oldest_user['content']} | 医生曾回复: {oldest_ai['content']}"
            doc_id = f"mem_{datetime.datetime.now().timestamp()}"
            
            self.long_term_db.add(
                documents=[archive_text],
                metadatas=[{"type": "historical_memory", "archive_date": str(datetime.date.today())}],
                ids=[doc_id]
            )

    def build_augmented_prompt(self, current_query, base_system_prompt):
        """
        组装终极 Prompt = System + 捞回的长期记忆 + 短期记忆窗口 + 当前提问
        """
        # 1. 触发时光机：用当前问题去 ChromaDB 捞取高度相关的远古记忆
        retrieved_memory = self.long_term_db.query(
            query_texts=[current_query],
            n_results=1,
            where={"type": "historical_memory"} # ⚠️ 元数据过滤：只在历史记忆池里搜
        )
        
        # 2. 解析捞出来的远古记忆（如果捞到了）
        long_term_context = ""
        if retrieved_memory['documents'] and retrieved_memory['documents'][0]:
            hit_text = retrieved_memory['documents'][0][0]
            # 作为补充背景，悄悄塞进系统提示词里
            long_term_context = f"\n[记忆外挂：患者在历史干预中曾提及：{hit_text}]\n"
            
        # 3. 组装终极 Prompt 队列
        final_messages = [{"role": "system", "content": base_system_prompt + long_term_context}]
        
        # 压入内存里的短期滑动窗口
        final_messages.extend(self.short_term_buffer)
        
        # 压入当前用户的最新提问
        final_messages.append({"role": "user", "content": current_query})
        
        return final_messages
