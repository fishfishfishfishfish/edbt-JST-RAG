"""TCRag 自定义检索器模块。

提供兼容 nuggetindex ``retriever_factory`` 的检索器实现。
每个检索器通过工厂函数 ``create_xxx_retriever(store) -> Retriever`` 注入,
Retriever 需实现
``async aretrieve(query, *, query_time, view, top_k, fusion, filters) -> list[RetrievalResult]``。
"""
