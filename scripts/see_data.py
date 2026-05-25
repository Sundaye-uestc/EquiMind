import chromadb
import os
import sys
from pathlib import Path

# chroma_db 默认路径（相对于项目根目录）
_CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "rag", "chroma_db")
CHROMA_DB_PATH = sys.argv[1] if len(sys.argv) > 1 else _CHROMA_DB_PATH

# 验证路径是否存在
if not os.path.exists(CHROMA_DB_PATH):
    print(f"错误：路径 {CHROMA_DB_PATH} 不存在！")
else:
    # 1. 连接到正确的 Chroma 数据库
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 2. 列出所有集合（确认是否存在）
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
    print("当前数据库中的所有集合：", collection_names)

    # 3. 只在集合存在时获取，避免报错
    if "agent" in collection_names:
        collection = client.get_collection(name="agent")

        # 4. 查询所有数据（关键修复：移除 include 中的 ids）
        # 说明：ids 是默认返回的，不需要手动加在 include 里
        all_data = collection.get(include=["embeddings", "metadatas", "documents"])

        # 5. 输出数据详情
        print(f"\n✅ 成功找到 agent 集合，共 {len(all_data['ids'])} 条数据")
        if all_data["ids"]:  # 有数据时才打印示例
            print("\n📌 第一条数据示例：")
            print(f"ID: {all_data['ids'][0]}")  # ids 仍能正常获取（默认返回）
            print(f"文档: {all_data['documents'][0][:100]}..." if all_data['documents'][0] else "无文档")
            print(f"元数据: {all_data['metadatas'][0]}")
            print(f"Embedding 长度: {len(all_data['embeddings'][0])}（前5个值：{all_data['embeddings'][0][:5]}）")
    else:
        print(f"\n❌ 未找到 agent 集合！当前存在的集合：{collection_names}")