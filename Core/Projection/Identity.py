"""Cross-representation identity contract for DIGIMON projections.

Every derived representation must reuse these producer identities verbatim
rather than reminting or normalizing them independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .FoundationIR import FoundationIR


@dataclass(frozen=True)
class ProjectionIdentityManifest:
    entity_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]
    predicate_ids: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    passage_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    namespace_ids: tuple[str, ...]
    source_registry_ids: tuple[str, ...]


def build_identity_manifest(ir: FoundationIR) -> ProjectionIdentityManifest:
    """Return the exact IDs downstream projections are required to preserve."""

    return ProjectionIdentityManifest(
        entity_ids=tuple(sorted(ir.entities_by_id)),
        assertion_ids=tuple(assertion.assertion_id for assertion in ir.assertions),
        predicate_ids=tuple(sorted({assertion.predicate for assertion in ir.assertions})),
        provenance_refs=tuple(sorted(ir.assertions_by_provenance_ref)),
        passage_ids=tuple(passage.passage_id for passage in ir.passages),
        source_refs=tuple(sorted({passage.source_ref for passage in ir.passages})),
        namespace_ids=tuple(
            sorted(
                {
                    value
                    for value in (
                        *(
                            assertion.namespace_id
                            for assertion in ir.assertions
                        ),
                        *(passage.namespace_id for passage in ir.passages),
                    )
                    if value is not None
                }
            )
        ),
        source_registry_ids=tuple(
            sorted(
                {
                    value
                    for value in (
                        *(
                            assertion.source_registry_id
                            for assertion in ir.assertions
                        ),
                        *(passage.source_registry_id for passage in ir.passages),
                    )
                    if value is not None
                }
            )
        ),
    )
