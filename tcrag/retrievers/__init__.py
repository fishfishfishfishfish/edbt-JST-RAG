"""TCRag custom retriever module.

Provides retriever implementations compatible with the nuggetindex
``retriever_factory``.
Each retriever is injected through the factory function
``create_xxx_retriever(store) -> Retriever``;
the Retriever must implement
``async aretrieve(query, *, query_time, view, top_k, fusion, filters) -> list[RetrievalResult]``.
"""
