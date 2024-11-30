# Build the PPR context for the Hipporag algorithm
await self._build_ppr_context()

# Augment the graph by ann searching
if self.config.enable_graph_augmentation:
    data_for_aug = {mdhash_id(node, prefix="ent-"): node for node in self.er_graph.graph.nodes()}
    await self._augment_graph(queries=data_for_aug)