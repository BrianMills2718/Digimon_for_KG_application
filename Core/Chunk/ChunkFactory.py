from typing import Any, List, Tuple
from Core.Common.Utils import mdhash_id
from collections import defaultdict
import json
from pathlib import Path
from loguru import logger

from Core.Schema.ChunkSchema import TextChunk


class ChunkingFactory:
    chunk_methods: dict = defaultdict(Any)

    def register_chunking_method(
        self,
        method_name: str,
        method_func=None,
    ):
        if self.has_chunk_method(method_name):
            return
        self.chunk_methods[method_name] = method_func

    def has_chunk_method(self, key: str) -> Any:
        return key in self.chunk_methods

    def get_method(self, key) -> Any:
        return self.chunk_methods.get(key)


CHUNKING_REGISTRY = ChunkingFactory()


def register_chunking_method(method_name):
    """Register a chunking method in the shared chunking registry."""

    def decorator(func):
        CHUNKING_REGISTRY.register_chunking_method(method_name, func)
        return func

    return decorator


def create_chunk_method(method_name):
    return CHUNKING_REGISTRY.get_method(method_name)


class ChunkFactory:
    """Load corpus documents and expose canonical DIGIMON text chunks.

    ``Corpus.json`` is document-oriented on the maintained MCP preparation path.
    Documents without an explicit ``chunk_id`` are therefore passed through the
    configured ``ChunkConfig`` strategy here. Existing corpora that already carry
    ``chunk_id`` values remain backward-compatible and are treated as pre-chunked.
    """

    def __init__(self, config):
        self.main_config = config
        self.workspaces = {}
        logger.info(
            f"ChunkFactory initialized with working_dir: {self.main_config.working_dir}"
        )

    def get_namespace(self, dataset_name, graph_type="er_graph"):
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty for creating a namespace.")

        class Namespace:
            def __init__(self, path):
                self.path = path

            def get_save_path(self, suffix=None):
                if suffix:
                    return str(Path(self.path) / suffix)
                return str(Path(self.path))

        working_dir_path = Path(self.main_config.working_dir) / dataset_name / graph_type
        data_root_path = Path(self.main_config.data_root) / dataset_name / graph_type

        if working_dir_path.exists():
            namespace_path = working_dir_path
        elif data_root_path.exists():
            namespace_path = data_root_path
        else:
            namespace_path = working_dir_path

        namespace_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ChunkFactory: Created/ensured namespace for dataset '{dataset_name}', "
            f"type '{graph_type}' at {namespace_path}"
        )
        return Namespace(str(namespace_path))

    def _corpus_paths(self, dataset_name: str) -> list[Path]:
        return [
            Path(self.main_config.working_dir) / dataset_name / "Corpus.json",
            Path(self.main_config.working_dir) / dataset_name / "corpus" / "Corpus.json",
            Path(self.main_config.data_root) / dataset_name / "Corpus.json",
        ]

    def _load_documents(self, corpus_path: Path) -> list[dict]:
        documents = []
        with open(corpus_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.error(
                        f"Error decoding JSON object on line {line_number} in "
                        f"{corpus_path}: {exc} - Line content: '{line[:100]}...'"
                    )
                    continue
                if not isinstance(value, dict):
                    logger.warning(
                        f"Skipping non-object corpus record on line {line_number}: "
                        f"{type(value).__name__}"
                    )
                    continue
                documents.append(value)
        return documents

    @staticmethod
    def _chunk_method_registered(method_name: str):
        method = create_chunk_method(method_name)
        if callable(method):
            return method

        # Registration is normally performed through Core.Chunk.__init__, but
        # importing ChunkFactory directly should still work in tests/tools.
        from Core.Chunk import Separator, Tokensize  # noqa: F401

        return create_chunk_method(method_name)

    async def _chunk_documents(self, documents: list[dict]) -> list[TextChunk]:
        from tiktoken import get_encoding

        encoding = get_encoding("cl100k_base")
        chunks: list[TextChunk] = []
        pending_docs = []

        # Preserve explicitly pre-chunked corpora exactly enough to remain
        # compatible with existing datasets/results.
        for position, doc in enumerate(documents):
            content = str(doc.get("content", "") or doc.get("context", "")).strip()
            if not content:
                continue

            if doc.get("chunk_id"):
                doc_id = str(doc.get("doc_id", f"doc_{position}"))
                chunks.append(
                    TextChunk(
                        tokens=int(doc.get("tokens") or len(encoding.encode(content))),
                        chunk_id=str(doc["chunk_id"]),
                        content=content,
                        doc_id=doc_id,
                        index=len(chunks),
                        title=doc.get("title", f"Document {doc_id}"),
                        metadata=dict(doc.get("metadata") or {}),
                    )
                )
                continue

            pending_docs.append((position, doc, content))

        if not pending_docs:
            return chunks

        chunk_config = getattr(self.main_config, "chunk", None)
        method_name = getattr(chunk_config, "chunk_method", "chunking_by_token_size")
        chunk_method = self._chunk_method_registered(method_name)
        if not callable(chunk_method):
            raise ValueError(f"Unknown or unregistered chunk method: {method_name}")

        max_tokens = max(1, int(getattr(chunk_config, "chunk_token_size", 1200) or 1200))
        overlap = max(0, int(getattr(chunk_config, "chunk_overlap_token_size", 100) or 0))
        if overlap >= max_tokens:
            logger.warning(
                f"Chunk overlap {overlap} must be smaller than chunk size {max_tokens}; "
                f"clamping overlap to {max_tokens - 1}."
            )
            overlap = max_tokens - 1

        doc_contents = [item[2] for item in pending_docs]
        doc_keys = [str(item[1].get("doc_id", f"doc_{item[0]}")) for item in pending_docs]
        title_list = [
            item[1].get("title", f"Document {doc_keys[index]}")
            for index, item in enumerate(pending_docs)
        ]
        tokens = encoding.encode_batch(doc_contents, num_threads=16)
        chunk_dicts = await chunk_method(
            tokens,
            doc_keys=doc_keys,
            tiktoken_model=encoding,
            title_list=title_list,
            overlap_token_size=overlap,
            max_token_size=max_tokens,
        )

        metadata_by_doc = {
            doc_key: dict(item[1].get("metadata") or {})
            for doc_key, item in zip(doc_keys, pending_docs)
        }
        per_doc_index: dict[str, int] = defaultdict(int)

        for chunk_dict in chunk_dicts or []:
            content = str(chunk_dict.get("content", "") or "").strip()
            if not content:
                continue
            doc_id = str(chunk_dict.get("doc_id", ""))
            local_index = per_doc_index[doc_id]
            per_doc_index[doc_id] += 1

            # Include source-document identity in the stable ID. Content-only
            # hashes collapse identical boilerplate appearing in different docs,
            # which destroys source provenance.
            chunk_id = mdhash_id(
                f"{doc_id}:{local_index}:{content}",
                prefix="chunk-",
            )
            metadata = dict(metadata_by_doc.get(doc_id, {}))
            metadata.update(
                {
                    "chunk_method": method_name,
                    "chunk_index_in_document": local_index,
                }
            )
            chunks.append(
                TextChunk(
                    tokens=int(chunk_dict.get("tokens") or len(encoding.encode(content))),
                    chunk_id=chunk_id,
                    content=content,
                    doc_id=doc_id,
                    index=len(chunks),
                    title=chunk_dict.get("title"),
                    metadata=metadata,
                )
            )

        return chunks

    async def get_chunks_for_dataset(self, dataset_name) -> List[Tuple[str, TextChunk]]:
        logger.info(f"ChunkFactory: Getting chunks for dataset '{dataset_name}'")

        corpus_path = next(
            (path for path in self._corpus_paths(dataset_name) if path.exists()),
            None,
        )
        if corpus_path is None:
            logger.error(f"Corpus file not found for dataset '{dataset_name}'")
            return []

        logger.info(f"Found corpus file at {corpus_path}")
        try:
            documents = self._load_documents(corpus_path)
            logger.info(
                f"Successfully loaded {len(documents)} document records from {corpus_path}"
            )
            chunks = await self._chunk_documents(documents)
            result = [(chunk.chunk_id, chunk) for chunk in chunks]
            logger.info(
                f"Created {len(result)} TextChunk objects for dataset '{dataset_name}' "
                f"from {len(documents)} document records"
            )
            return result
        except Exception as exc:
            logger.exception(f"Error loading/chunking corpus '{dataset_name}': {exc}")
            return []
