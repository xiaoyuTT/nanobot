# nanobot/agent/tools/rag_tool.py

from typing import TYPE_CHECKING, Any
from pathlib import Path
from loguru import logger
from nanobot.agent.tools.base import Tool

class SearchKnowledgeTool(Tool):
    """检索知识库工具"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    @property
    def name(self) -> str:
        return "search_knowledge"

    @property
    def description(self) -> str:
        return """Search the indexed knowledge base for relevant information.
            Use this tool when you need to find information from previously indexed documents.
            The search uses hybrid retrieval (vector similarity + BM25) for best results.

            Example queries:
            - "What is reinforcement learning?"
            - "How does RAG work?"
            - "Explain the difference between supervised and unsupervised learning"
            """
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, top_k: int = 3) -> str:
        """
        执行检索

        返回：
        - 格式化的检索结果字符串
        """
        try:
            # 调用 RAG 引擎检索
            results = await self.rag_engine._search(query, top_k=top_k)

            if not results:
                return "No relevant information found in the knowledge base."

            # 格式化结果
            output_parts = [f"Found {len(results)} relevant documents:\n"]

            for i, doc in enumerate(results, 1):
                content = doc.page_content[:300]  # 限制每个结果的长度
                source = doc.metadata.get("source", "unknown")
                filename = Path(source).name if source != "unknown" else "unknown"

                output_parts.append(
                    f"[{i}] From: {filename}\n{content}...\n"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error(f"Search knowledge failed: {e}")
            return f"Error searching knowledge base: {str(e)}"


class IndexDocumentsTool(Tool):
    """索引文档工具"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    @property
    def name(self) -> str:
        return "index_documents"

    @property
    def description(self) -> str:
        return """Index documents into the knowledge base for future retrieval.

            This tool loads documents, splits them into chunks, and adds them to the vector database.
            Supported formats: Markdown (.md), Text (.txt)

            When to use:
            - When you have new documents to add to the knowledge base
            - When you want to update the knowledge base with additional information
            - When user provides new files to be indexed

            Example usage:
            - Index a single file: paths=["/path/to/document.md"]
            - Index multiple files: paths=["/path/to/doc1.md", "/path/to/doc2.txt"]
        """
    
    @property    
    def parameters(self) -> dict[str, Any]:
        return {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    "description": "List of file paths to index"
                }
            },
            "required": ["paths"]
        }

    async def execute(self, paths: list[str]) -> str:
        """
        执行索引

        返回：
        - 索引结果摘要
        """
        try:
            # 转换为 Path 对象
            file_paths = [Path(p).expanduser() for p in paths]

            # 验证文件存在
            valid_paths = []
            for fp in file_paths:
                if not fp.exists():
                    logger.warning(f"File not found: {fp}")
                    continue
                if not fp.is_file():
                    logger.warning(f"Not a file: {fp}")
                    continue
                valid_paths.append(fp)

            if not valid_paths:
                return "No valid files to index."

            # 调用 RAG 引擎索引
            result = await self.rag_engine.add_documents(valid_paths)

            # 格式化返回结果
            output = [
                f"Indexing completed:",
                f"- Successfully indexed: {result['success']} files",
                f"- Failed: {result['failed']} files",
                f"- Total chunks created: {result['total_chunks']}",
                f"- Indexed files: {', '.join([Path(f).name for f in result['indexed_files']])}"
            ]

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Index documents failed: {e}")
            return f"Error indexing documents: {str(e)}"


class ListKnowledgeBaseTool(Tool):
    """查看知识库信息工具"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    @property
    def name(self) -> str:
        return "list_knowledge_base"

    @property
    def description(self) -> str:
        return """
            List information about the current knowledge base.
            Shows statistics including:
            - Total number of indexed document chunks
            - Number of indexed files
            - List of indexed file names
        """
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }

    async def execute(self) -> str:
        """
        获取知识库统计信息
        """
        try:
            stats = self.rag_engine.get_stats()
            indexed_files = self.rag_engine.list_indexed_files()

            if stats['total_chunks'] == 0:
                return "Knowledge base is empty. Use 'index_documents' to add documents."

            output = [
                "Knowledge Base Statistics:",
                f"- Total chunks: {stats['total_chunks']}",
                f"- Indexed files: {stats['indexed_files']}",
                f"\nIndexed files:"
            ]

            for i, file_path in enumerate(indexed_files, 1):
                filename = Path(file_path).name
                output.append(f"  {i}. {filename}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"List knowledge base failed: {e}")
            return f"Error getting knowledge base info: {str(e)}"