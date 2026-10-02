from app.data.devices import DEVICE_DOCUMENTS
from app.utils.bm25_index import sparse_search
from app.utils.embeddings import dense_search

def rrf(rankings, k=60):
    """rankings: list of ranked lists of doc indices"""
    fused = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(query: str, final_k: int = 5, candidate_k: int = 50) -> list[tuple[str, float]]:
    sparse_results=sparse_search(query,top_k=candidate_k)
    dense_results=dense_search(query,top_k=candidate_k)
    dense_indices=[DEVICE_DOCUMENTS.index(doc[0]) for doc in dense_results]
    sparse_indices=[DEVICE_DOCUMENTS.index(doc) for doc in sparse_results]
    fused_results=rrf([sparse_indices,dense_indices])
    return [(DEVICE_DOCUMENTS[index], score) for index, score in fused_results[:final_k]]


