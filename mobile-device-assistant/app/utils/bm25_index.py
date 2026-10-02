import string

from rank_bm25 import BM25Okapi

from app.data.devices import DEVICE_DOCUMENTS


def _tokenize(text: str) -> list[str]:
	return text.translate(str.maketrans("", "", string.punctuation)).lower().split()


_BM25_INDEX = BM25Okapi([_tokenize(document) for document in DEVICE_DOCUMENTS])


def sparse_search(query: str, top_k: int = 5) -> list[str]:
	tokenized_query = _tokenize(query)
	scores = _BM25_INDEX.get_scores(tokenized_query)
	top_k_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
	return [DEVICE_DOCUMENTS[index] for index in top_k_indices]
