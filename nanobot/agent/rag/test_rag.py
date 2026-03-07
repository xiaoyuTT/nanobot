# test_rag_multi_docs.py
import asyncio
from pathlib import Path
from nanobot.agent.rag.engine import RAGEngine

async def test():
    # 1. 初始化（不绑定文档）
    engine = RAGEngine(workspace=Path("/home/wangtao/wt/mcp-servers/nanobot"))

    # 2. 索引多个文档
    test_files = [
        Path("/home/wangtao/wt/mcp-servers/nanobot/nanobot/agent/rag/data/test.md"),
        # 可以添加更多文件
    ]

    result = await engine.add_documents(test_files)
    print(f"✅ 索引结果: {result}")

    # 3. 查看统计
    stats = engine.get_stats()
    print(f"📊 统计信息: {stats}")

    # 4. 测试检索
    results = await engine._search("强化学习", top_k=2)
    print(f"\n📄 检索到 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.page_content[:100]}...")

    # 5. 测试上下文生成
    context = await engine.get_doc_context("什么是RAG", max_tokens=500)
    print(f"\n📝 上下文 ({len(context)} 字符):\n{context[:300]}...")

    # 6. 添加更多文档（增量索引）
    # more_files = [Path("another_doc.md")]
    # result2 = await engine.add_documents(more_files)
    # print(f"✅ 增量索引: {result2}")

asyncio.run(test())