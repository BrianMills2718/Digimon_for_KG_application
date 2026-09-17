import asyncio
from typing import Any, List

from Core.Common.Logger import logger
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.DelimiterExtraction import DelimiterExtractionMixin
from Core.Storage.NetworkXStorage import NetworkXStorage


class RKGraph(DelimiterExtractionMixin, BaseGraph):
    """Relationship-keyword graph built from delimiter extraction."""

    def __init__(self, config, llm, encoder):
        from Core.Common.TokenizerWrapper import TokenizerWrapper

        tokenizer = TokenizerWrapper()
        super().__init__(config, llm, tokenizer)
        self._graph = NetworkXStorage()

        # Handle both full Config and GraphConfig inputs. RK's defining feature
        # is keyword-enriched relationships, so enable the keyword extraction
        # prompt explicitly instead of relying on a caller-side override.
        self.graph_config = config.graph if hasattr(config, "graph") else config
        self.graph_config.enable_edge_keywords = True
        self.encoder = encoder

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, "TextChunk"]):
        chunk_key, chunk_info = chunk_key_pair
        records = await self._extract_records_from_chunk(chunk_info)
        return await self._build_graph_from_records(records, chunk_key)

    async def _build_graph(self, chunk_list: List[Any]):
        try:
            elements = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list]
            )
            await self.__graph__(elements)
            if self.node_num <= 0:
                logger.error(
                    "RKGraph extraction completed without producing any graph nodes; "
                    "treating the build as failed."
                )
                return False
            return True
        except Exception as exc:
            logger.exception(f"Error building graph: {exc}")
            return False
        finally:
            logger.info("Constructing graph finished")

    @property
    def entity_metakey(self):
        return "entity_name"
