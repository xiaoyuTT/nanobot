from pathlib import Path
from loguru import logger

class RAGEngine:
    """A simple RAG engine"""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.persist_dir = workspace / "rag" / "vectorstore"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 多文档支持：存储所有已索引的文档片段
        self._all_chunks = []  # 用于 BM25 检索
        self._indexed_files = set()  # 追踪已索引的文件路径
        logger.info(f"RAG Engine initialized with workspace: {self.workspace}")

        self._embeddings = None
        self._vectorstore = None
        self._vector_retriever = None
        self._bm25_retriever = None

    def _ensure_initialized(self):

      if self._embeddings is None:
          from langchain_huggingface import HuggingFaceEmbeddings
          self._embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

      if self._vectorstore is None:
          from langchain_chroma import Chroma
          self._vectorstore = Chroma(
              collection_name="documents",
              embedding_function=self._embeddings,
              persist_directory=str(self.persist_dir)
          )
          logger.info(f"VectorStore initialized at {self.persist_dir}")

      if self._vector_retriever is None:
          self._vector_retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

    async def _search(self, query: str, top_k: int = 5):
      """
      混合检索（向量 + BM25）
   
      参数：
      - query: 查询文本
      - top_k: 返回结果数量
   
      返回：
      - List[Document]: 去重后的文档列表
      """
      self._ensure_initialized()

      if not self._all_chunks:
          logger.warning("No documents indexed yet. Please call add_documents() first.")
          return []

      try:
          # 1. 向量检索（使用 invoke 方法 - Langchain 新版 API）
          vector_results = self._vector_retriever.invoke(query)
          logger.info(f"向量检索返回 {len(vector_results)} 条结果")

          # 2. BM25 检索（如果存在）
          bm25_results = []
          if self._bm25_retriever:
              bm25_results = self._bm25_retriever.invoke(query)
              logger.info(f"BM25检索返回 {len(bm25_results)} 条结果")

          # 3. 合并去重
          seen = set()
          combined = []

          for doc in vector_results + bm25_results:
              content_hash = hash(doc.page_content)
              if content_hash not in seen:
                  seen.add(content_hash)
                  combined.append(doc)

          # 4. 返回前 top_k 个结果
          return combined[:top_k]

      except Exception as e:
          logger.error(f"检索过程中发生错误: {e}")
          return []

    def get_stats(self) -> dict:
      """获取 RAG 系统统计信息"""
      self._ensure_initialized()

      vector_count = 0
      try:
          # Chroma 可能没有 count() 方法，使用其他方式
          vector_count = len(self._all_chunks)
      except:
          pass

      return {
          "total_chunks": len(self._all_chunks),
          "indexed_files": len(self._indexed_files),
          "vector_store_chunks": vector_count,
          "persist_directory": str(self.persist_dir)
      }

    def list_indexed_files(self) -> list[str]:
        """列出所有已索引的文件"""
        return list(self._indexed_files)

    async def clear_index(self):
        """清空所有索引数据"""
        self._ensure_initialized()

        # 清空向量库（需要重建）
        import shutil
        shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 重置状态
        self._all_chunks = []
        self._indexed_files = set()
        self._vectorstore = None
        self._vector_retriever = None
        self._bm25_retriever = None

        logger.info("Index cleared")

    async def add_documents(self, file_paths: list[Path]) -> dict:
      """
      索引多个文档到 RAG 系统
   
      参数：
      - file_paths: 文档路径列表
   
      返回：
      - dict: 索引结果统计
          {
              "success": 成功数,
              "failed": 失败数,
              "total_chunks": 总片段数,
              "indexed_files": 已索引文件列表
          }
      """
      self._ensure_initialized()

      success_count = 0
      failed_count = 0
      new_chunks = []

      for file_path in file_paths:
          # 跳过已索引的文件
          file_str = str(file_path.absolute())
          if file_str in self._indexed_files:
              logger.info(f"Skipping already indexed file: {file_path.name}")
              continue

          try:
              # 1. 加载文档
              from langchain_community.document_loaders import UnstructuredMarkdownLoader
              docs = UnstructuredMarkdownLoader(str(file_path)).load()

              # 2. 分块
              from langchain_text_splitters import RecursiveCharacterTextSplitter
              text_splitter = RecursiveCharacterTextSplitter(
                  chunk_size=1000,
                  chunk_overlap=200,
                  add_start_index=True
              )
              chunks = text_splitter.split_documents(docs)

              # 3. 累积新片段
              new_chunks.extend(chunks)
              self._indexed_files.add(file_str)
              success_count += 1

              logger.info(f"Loaded {file_path.name}: {len(chunks)} chunks")

          except Exception as e:
              logger.error(f"Failed to load {file_path}: {e}")
              failed_count += 1

      # 4. 批量添加到向量库
      if new_chunks:
          self._vectorstore.add_documents(new_chunks)
          self._all_chunks.extend(new_chunks)

          # 5. 重建 BM25 检索器（基于所有文档）
          from langchain_community.retrievers import BM25Retriever
          self._bm25_retriever = BM25Retriever.from_documents(self._all_chunks)

          logger.info(f"Added {len(new_chunks)} chunks to vector store")
          logger.info(f"Total indexed chunks: {len(self._all_chunks)}")

      return {
          "success": success_count,
          "failed": failed_count,
          "total_chunks": len(new_chunks),
          "indexed_files": list(self._indexed_files)
      }

    async def get_doc_context(self, query: str, max_tokens: int = 1500) -> str:
        """
        获取格式化的上下文（用于注入 LLM prompt）
    
        参数：
        - query: 查询文本
        - max_tokens: 最大 token 数（粗略估算：1 token ≈ 1.5 字符）
    
        返回：
        - str: 格式化的上下文文本
        """
        self._ensure_initialized()
        
        # 1. 检索相关文档
        results = await self._search(query, top_k=5)

        if not results:
            return ""

        # 2. 格式化为可读文本
        context_parts = []
        total_chars = 0
        max_chars = int(max_tokens * 1.5)  # 粗略估算

        for i, doc in enumerate(results, 1):
            # 获取文档内容和元数据
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")

            # 格式化
            formatted = f"[文档 {i} - 来源: {source}]\n{content}\n"

            # 检查是否超出限制
            if total_chars + len(formatted) > max_chars:
                # 截断最后一个文档
                remaining = max_chars - total_chars
                if remaining > 100:  # 至少保留 100 字符
                    formatted = formatted[:remaining] + "\n...(已截断)"
                    context_parts.append(formatted)
                break

            context_parts.append(formatted)
            total_chars += len(formatted)

        # 3. 组合成最终上下文
        context = "\n---\n\n".join(context_parts)
        return context