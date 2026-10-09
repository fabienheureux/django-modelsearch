"""Pluggable embedding providers for semantic search.

This module provides a simple interface for generating text embeddings
that can be used for semantic (vector) search. The default provider is
a deterministic hash-based embedding that requires no external
dependencies, making it suitable for development and testing.

For production use, you can configure a real embedding model by setting
the ``MODELSEARCH_EMBEDDING_PROVIDER`` setting, e.g.::

    MODELSEARCH_EMBEDDING_PROVIDER = "modelsearch.embeddings.SentenceTransformersProvider"
    MODELSEARCH_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    MODELSEARCH_EMBEDDING_DIMENSIONS = 384

Or implement your own provider by subclassing ``BaseEmbeddingProvider``.
"""

import hashlib
import importlib
import logging

from abc import ABC, abstractmethod
from collections.abc import Sequence

from django.conf import settings


logger = logging.getLogger(__name__)

# Default embedding dimensions for the hash-based provider.
# This is intentionally small for speed; real providers will use 384-1536.
DEFAULT_HASH_DIMENSIONS = 128


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement :meth:`embed_texts` which takes a list of
    strings and returns a list of embedding vectors (lists of floats).
    """

    #: The dimensionality of the embedding vectors produced by this provider.
    dimensions: int = 0

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text. Each vector
            is a list of floats of length ``self.dimensions``.
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Default implementation calls :meth:`embed_texts` with a single-item list.
        Override for providers that have a more efficient single-text API.
        """
        return self.embed_texts([text])[0]


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic hash-based embedding provider.

    Uses SHA-256 hashing of character n-grams to produce a fixed-dimension
    embedding vector. This is **not** a real semantic embedding — it only
    captures lexical similarity (similar words produce similar vectors).
    It is useful for:

    - Development and testing without external dependencies
    - Bootstrapping the semantic search infrastructure
    - Fallback when a real embedding provider is unavailable

    For production semantic search, use a real embedding model.
    """

    def __init__(self, dimensions: int | None = None):
        self.dimensions = dimensions or DEFAULT_HASH_DIMENSIONS

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        """Generate a deterministic embedding for a single text.

        The embedding is a bag-of-ngrams hashed into a fixed-dimension vector,
        L2-normalized so that cosine similarity is a simple dot product.
        """
        import math

        # Normalize text: lowercase, strip accents (basic), split into words
        normalized = text.lower().strip()
        if not normalized:
            return [0.0] * self.dimensions

        # Create character n-grams (3-grams and 4-grams) for sub-word similarity
        ngrams = set()
        padded = f" {normalized} "
        for n in (3, 4):
            for i in range(len(padded) - n + 1):
                ngrams.add(padded[i : i + n])

        # Hash each n-gram into the vector space
        vector = [0.0] * self.dimensions
        for ngram in ngrams:
            # Use SHA-256 for deterministic hashing
            digest = hashlib.sha256(ngram.encode("utf-8")).digest()
            # Use first 4 bytes for index, next byte for sign
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Embedding provider using sentence-transformers (local models).

    Requires the ``sentence-transformers`` package::

        pip install sentence-transformers

    Configure via settings::

        MODELSEARCH_EMBEDDING_PROVIDER = "modelsearch.embeddings.SentenceTransformersProvider"
        MODELSEARCH_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or getattr(
            settings,
            "MODELSEARCH_EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._model = None
        # Dimensions will be set after loading the model
        self.dimensions = 0

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformersProvider. "
                    "Install it with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)
            # Use get_embedding_dimension() (newer API) with fallback to
            # get_sentence_embedding_dimension() for older versions
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimensions = self._model.get_embedding_dimension()
            else:
                self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Requires the ``openai`` package and an API key::

        pip install openai

    Configure via settings::

        MODELSEARCH_EMBEDDING_PROVIDER = "modelsearch.embeddings.OpenAIEmbeddingProvider"
        MODELSEARCH_EMBEDDING_MODEL = "text-embedding-3-small"
        MODELSEARCH_OPENAI_API_KEY = "sk-..."
    """

    # OpenAI embedding dimensions by model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or getattr(
            settings, "MODELSEARCH_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.api_key = api_key or getattr(settings, "MODELSEARCH_OPENAI_API_KEY", None)
        self.dimensions = self.MODEL_DIMENSIONS.get(self.model_name, 1536)
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai is required for OpenAIEmbeddingProvider. "
                    "Install it with: pip install openai"
                ) from e
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=list(texts),
            model=self.model_name,
        )
        return [item.embedding for item in response.data]


# Registry of available providers
_PROVIDERS: dict[str, type[BaseEmbeddingProvider]] = {
    "hash": HashEmbeddingProvider,
    "sentence-transformers": SentenceTransformersProvider,
    "openai": OpenAIEmbeddingProvider,
}

# Cache for the singleton provider instance
_provider_instance: BaseEmbeddingProvider | None = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Get the configured embedding provider.

    Reads ``MODELSEARCH_EMBEDDING_PROVIDER`` from Django settings.
    Can be either:

    - A provider key: ``"hash"``, ``"sentence-transformers"``, ``"openai"``
    - A dotted Python path: ``"myapp.embeddings.MyCustomProvider"``

    Defaults to the hash-based provider if not configured.
    """
    global _provider_instance

    if _provider_instance is not None:
        return _provider_instance

    provider_path = getattr(
        settings,
        "MODELSEARCH_EMBEDDING_PROVIDER",
        "hash",
    )

    # Check if it's a registered provider key
    if provider_path in _PROVIDERS:
        provider_class = _PROVIDERS[provider_path]
    else:
        # Try to import as a dotted path
        try:
            module_path, class_name = provider_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning(
                "Could not load embedding provider '%s': %s. "
                "Falling back to hash-based provider.",
                provider_path,
                e,
            )
            provider_class = HashEmbeddingProvider

    _provider_instance = provider_class()
    logger.info(
        "Using embedding provider: %s (dimensions=%d)",
        type(_provider_instance).__name__,
        _provider_instance.dimensions,
    )
    return _provider_instance


def reset_embedding_provider() -> None:
    """Reset the cached provider instance.

    Useful for testing or when settings change at runtime.
    """
    global _provider_instance
    _provider_instance = None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are assumed to be L2-normalized (as produced by our providers),
    so this is just a dot product. Falls back to full computation if not normalized.
    """
    import math

    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions must match: {len(vec_a)} != {len(vec_b)}"
        )

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
