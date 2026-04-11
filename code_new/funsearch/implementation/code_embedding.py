"""Utilities for converting code strings into reduced embedding vectors."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


_EMBEDDER_CACHE: dict[tuple[str, str | None], "CodeEmbedder"] = {}


class CodeEmbeddingError(RuntimeError):
	"""Raised when code embedding or PCA reduction fails."""


class CodeEmbedder:
	"""Embeds code strings and reduces vectors to a fixed PCA dimension."""

	def __init__(self, model: str, hf_token: str | None = None) -> None:
		self._model = model
		self._hf_token = hf_token
		try:
			self._encoder = SentenceTransformer(model, token=hf_token)
		except Exception as exc:
			raise CodeEmbeddingError(f'Failed to load embedding model `{model}`: {exc}') from exc

	def embed_code(self, code: str) -> np.ndarray:
		"""Returns the raw embedding vector for one code string."""
		if not code or not code.strip():
			raise ValueError('`code` must be a non-empty string.')

		try:
			vector = self._encoder.encode(code, convert_to_numpy=True)
		except Exception as exc:
			raise CodeEmbeddingError(f'Failed to embed code with `{self._model}`: {exc}') from exc

		vector = np.asarray(vector, dtype=np.float32)
		if vector.ndim != 1 or vector.size == 0:
			raise CodeEmbeddingError('Embedding response has invalid shape.')
		return vector

	@staticmethod
	def _reduce_with_pca(vectors: np.ndarray, out_dim: int) -> np.ndarray:
		"""Runs PCA using SVD and returns projected vectors."""
		if vectors.ndim != 2:
			raise ValueError('`vectors` must be a 2D array.')
		if out_dim <= 0:
			raise ValueError('`out_dim` must be positive.')

		num_samples, num_features = vectors.shape
		if out_dim > num_features:
			raise ValueError('`out_dim` cannot exceed embedding dimension.')

		max_rank = min(num_samples - 1, num_features)
		if max_rank <= 0:
			single = vectors[0]
			if out_dim <= single.size:
				return single[:out_dim][None, :]
			padded = np.pad(single, (0, out_dim - single.size))
			return padded[None, :]

		work_dim = min(out_dim, max_rank)
		centered = vectors - np.mean(vectors, axis=0, keepdims=True)
		_, _, vt = np.linalg.svd(centered, full_matrices=False)
		components = vt[:work_dim].T
		reduced = centered @ components

		if reduced.shape[1] < out_dim:
			reduced = np.pad(reduced, ((0, 0), (0, out_dim - reduced.shape[1])))
		return reduced.astype(np.float32, copy=False)

	def embed_and_reduce(
			self,
			code: str,
			pca_dim: int,
			reference_codes: Sequence[str] | None = None,
	) -> np.ndarray:
		"""Embeds `code` and returns a PCA-reduced vector of length `pca_dim`.

		Args:
			code: Target code string to encode.
			pca_dim: Output vector dimension after PCA.
			reference_codes: Optional additional code samples used to fit PCA basis.
					If omitted, a single-sample fallback is used.

		Returns:
			A 1D numpy vector with shape `(pca_dim,)`.
		"""
		target_vector = self.embed_code(code)

		vectors = [target_vector]
		if reference_codes:
			for ref_code in reference_codes:
				if ref_code and ref_code.strip():
					vectors.append(self.embed_code(ref_code))

		stacked = np.vstack(vectors)
		reduced = self._reduce_with_pca(stacked, out_dim=pca_dim)
		return reduced[0]


def _load_embedding_config() -> tuple[str, int, str | None]:
	env_path = Path(__file__).resolve().parents[2] / '.env'
	load_dotenv(dotenv_path=env_path)

	model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
	pca_dim_raw = os.getenv('EMBEDDING_PCA_DIM', '16')
	hf_token = os.getenv('HF_TOKEN', '').strip() or os.getenv('HUGGINGFACE_HUB_TOKEN', '').strip() or None
	try:
		pca_dim = int(pca_dim_raw)
	except ValueError as exc:
		raise ValueError('EMBEDDING_PCA_DIM must be an integer.') from exc
	if pca_dim <= 0:
		raise ValueError('EMBEDDING_PCA_DIM must be positive.')

	return model, pca_dim, hf_token


def _get_embedder(model: str, hf_token: str | None) -> CodeEmbedder:
	cache_key = (model, hf_token)
	embedder = _EMBEDDER_CACHE.get(cache_key)
	if embedder is None:
		embedder = CodeEmbedder(model=model, hf_token=hf_token)
		_EMBEDDER_CACHE[cache_key] = embedder
	return embedder


def embed_code_to_16d(
		code: str,
		reference_codes: Sequence[str] | None = None,
		model: str | None = None,
		pca_dim: int | None = None,
) -> np.ndarray :
	"""Convenience helper: code string -> embedding -> PCA-reduced vector."""
	env_model, env_pca_dim, env_hf_token = _load_embedding_config()
	selected_model = model or env_model
	selected_pca_dim = pca_dim if pca_dim is not None else env_pca_dim

	embedder = _get_embedder(model=selected_model, hf_token=env_hf_token)
	return embedder.embed_and_reduce(
			code=code,
			pca_dim=selected_pca_dim,
			reference_codes=reference_codes,
	)
