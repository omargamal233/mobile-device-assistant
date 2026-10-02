from functools import lru_cache

import numpy as np

from app.data.devices import DEVICE_DOCUMENTS


@lru_cache(maxsize=1)
def _get_model():
	from sentence_transformers import SentenceTransformer

	return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_document_embeddings() -> np.ndarray:
	return _get_model().encode(
		DEVICE_DOCUMENTS,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)


def dense_search(query: str, top_k: int = 5) -> list[tuple[str, float]]:
	document_embeddings = _get_document_embeddings()
	query_embedding = _get_model().encode(
		[query],
		convert_to_numpy=True,
		normalize_embeddings=True,
	)[0]
	scores = np.dot(document_embeddings, query_embedding)
	top_k_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
	return [(DEVICE_DOCUMENTS[index], float(scores[index])) for index in top_k_indices]
