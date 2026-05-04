"""
Retrieval Module — LTE Inference Phase
========================================
Implements the memory bank retrieval used during LTE inference.
At query time, retrieve the most relevant edit descriptor from the
memory bank and prepend it to the prompt.

Paper: "a retrieval mechanism pulls the most relevant edit descriptor
from a memory bank at query time"

We implement BM25 retrieval (fast, no GPU needed).
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi


class EditMemoryBank:
    """
    BM25-based memory bank for LTE inference phase.

    Usage:
        bank = EditMemoryBank(edit_descriptors)
        edit, score = bank.retrieve(query)
        prompt = f"[Updated Information]{edit} [Query]{query}"
    """

    def __init__(self, edit_descriptors: List[str]):
        """
        Args:
            edit_descriptors: list of edit fact strings
                e.g. ["The capital of France is Berlin", ...]
        """
        self.edit_descriptors = edit_descriptors
        tokenized = [doc.lower().split() for doc in edit_descriptors]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[Retrieval] Memory bank built with {len(edit_descriptors)} entries")

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant edit descriptors for a query.

        Args:
            query : the question being asked
            top_k : number of results to return

        Returns:
            list of (edit_descriptor, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            (self.edit_descriptors[i], float(scores[i]))
            for i in top_indices
        ]

    def retrieve_top1(self, query: str) -> str:
        """Return the single best matching edit descriptor."""
        results = self.retrieve(query, top_k=1)
        return results[0][0] if results else ""

    def format_lte_prompt(self, query: str) -> str:
        """
        Full LTE inference prompt:
        retrieve edit + format with paper's template.
        """
        edit = self.retrieve_top1(query)
        return f"[Updated Information]{edit} [Query]{query}"
