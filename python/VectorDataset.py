import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class VectorDataset:
    @dataclass
    class Row:
        id: int
        text: str
        embedding: List[float]
        label: str
        metadata: str = ""
        timestamp: float = -1

        def to_arrow(self) -> List[pa.array]:
            return [
                pa.array(self.id, type=pa.int64()),
                pa.array(self.text, type=pa.string()),
                pa.array(
                    self.embedding, type=pa.list_(pa.float32())
                ),  # Remove list size constraint
                pa.array(self.label, type=pa.string()),
                pa.array(self.metadata, type=pa.string()),
                pa.array(self.timestamp, type=pa.timestamp("us")),
            ]

        def to_list(self):
            return {
                "text": self.text,
                "embedding": self.embedding,
                "label": self.label,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
            }

    """
    Vector dataset with Arrow backend optimized for FAISS integration.
    Supports efficient row addition, querying, and zero-copy vector extraction.
    """

    def __init__(self, name: str, base_path: str = ".", embedding_dim: int = 768):
        self.name = name
        self.base_path = Path(base_path)
        self.embedding_dim = embedding_dim
        self.parquet_path = self.base_path / f"{name}.parquet"
        self.metadata_path = self.base_path / f"{name}_meta.json"

        # Create directories if needed
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self.schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("doc_id", pa.int64()),
                pa.field("text", pa.string()),
                pa.field(
                    "embedding", pa.list_(pa.float32())
                ),  # Remove fixed size constraint
                pa.field("label", pa.string()),
                pa.field("metadata", pa.string()),  # JSON string for flexible metadata
                pa.field("timestamp", pa.timestamp("us")),
            ]
        )

        # Load existing data or create new
        self.table = self._load_or_create()
        self._next_id = self._get_next_id()
        self._next_doc_id = self._get_next_doc_id()

        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product

    def _load_or_create(self) -> pa.Table:
        """Load existing parquet file or create empty table."""
        if self.parquet_path.exists():
            return pq.read_table(self.parquet_path)
        else:
            # Create empty arrays for each field in schema
            empty_arrays = []
            for field in self.schema:
                if field.type == pa.int64():
                    empty_arrays.append(pa.array([], type=pa.int64()))
                elif field.type == pa.string():
                    empty_arrays.append(pa.array([], type=pa.string()))
                elif field.type == pa.list_(pa.float32()):
                    empty_arrays.append(pa.array([], type=pa.list_(pa.float32())))
                elif field.type == pa.timestamp("us"):
                    empty_arrays.append(pa.array([], type=pa.timestamp("us")))
                else:
                    # Fallback for any other types
                    empty_arrays.append(pa.array([], type=field.type))

            return pa.table(empty_arrays, schema=self.schema)

    def _get_next_id(self) -> int:
        """Get next available ID."""
        if len(self.table) == 0:
            return 1
        return pc.max(self.table["id"]).as_py() + 1

    def _get_next_doc_id(self) -> int:
        """Get next available document ID."""
        if len(self.table) == 0:
            return 1
        return pc.max(self.table["doc_id"]).as_py() + 1

    def add_doc(
        self,
        text: str,
        embedding: np.ndarray,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a single row to the dataset."""
        return self.add_batch(
            [
                {
                    "text": text,
                    "embedding": embedding,
                    "label": label,
                    "metadata": metadata or {},
                }
            ]
        )[0]

    def add_docs(self, rows: List[List[Dict[str, Any]]]) -> List[int]:
        """Add multiple rows efficiently from single document."""
        if not rows:
            return []

        # Prepare batch data
        ids = []
        doc_ids = []
        texts = []
        embeddings = []
        labels = []
        metadatas = []
        timestamps = []

        current_time = time.time()

        for doc in rows:
            t_start = time.time()
            for row in doc:
                # Validate embedding dimension
                emb = np.array(row["embedding"], dtype=np.float32)
                if emb.shape[0] != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb.shape[0]}"
                    )

                ids.append(self._next_id)
                doc_ids.append(self._next_doc_id)
                texts.append(row["text"])
                embeddings.append(emb.tolist())
                labels.append(row.get("label", ""))
                metadatas.append(json.dumps(row.get("metadata", {})))
                timestamps.append(current_time)
                self._next_id += 1
            t_end = time.time()
            self._next_doc_id += 1

        # Create Arrow arrays with explicit schema matching
        arrays = [
            pa.array(ids, type=pa.int64()),
            pa.array(doc_ids, type=pa.int64()),
            pa.array(texts, type=pa.string()),
            pa.array(
                embeddings, type=pa.list_(pa.float32())
            ),  # Remove list size constraint
            pa.array(labels, type=pa.string()),
            pa.array(metadatas, type=pa.string()),
            pa.array(timestamps, type=pa.timestamp("us")),
        ]

        # Create table with arrays in schema order
        new_table = pa.table(arrays, schema=self.schema)

        new_ind = list(
            range(self.table.num_rows, self.table.num_rows + new_table.num_rows)
        )
        # Concatenate with existing table
        self.table = pa.concat_tables([self.table, new_table])

        # update faiss
        new_emb = self.get_embeddings_for_faiss(new_ind)
        self.index.add(new_emb)

        return ids

    def get_embeddings_for_faiss(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get embeddings as numpy array for FAISS (zero-copy when possible).

        Args:
            indices: Specific row indices to extract, or None for all

        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        if indices is not None:
            # Filter table by indices
            filtered_table = self.table.take(indices)
            embedding_column = filtered_table["embedding"]
        else:
            embedding_column = self.table["embedding"]

        # Convert Arrow list array to numpy
        # This creates a zero-copy view when possible
        embeddings_list = embedding_column.to_pylist()
        return np.array(embeddings_list, dtype=np.float32)

    def filter(self, condition) -> "VectorDataset":
        """Filter dataset based on condition."""
        filtered_table = self.table.filter(condition)

        # Create new dataset instance with filtered data
        new_dataset = VectorDataset(
            f"{self.name}_filtered", str(self.base_path), self.embedding_dim
        )
        new_dataset.table = filtered_table
        new_dataset._next_id = self._next_id

        return new_dataset

    def query_by_text(self, text_pattern: str) -> "VectorDataset":
        """Query rows containing text pattern."""
        condition = pc.match_substring_regex(self.table["text"], text_pattern)
        return self.filter(condition)

    def query_by_label(self, label: str) -> "VectorDataset":
        """Query rows by exact label match."""
        condition = pc.equal(self.table["label"], label)
        return self.filter(condition)

    def query_by_doc_id(self, doc_id: List[int]) -> "VectorDataset":
        """Query rows by document ID."""
        condition = pc.is_in(self.table["doc_id"], pa.array(doc_id))
        return self.filter(condition)

    def get_by_ids(self, ids: List[int]) -> Dict[str, Any]:
        """Get rows by their IDs."""
        condition = pc.is_in(self.table["id"], pa.array(ids))
        filtered_table = self.table.filter(condition)
        return self._table_to_dict(filtered_table)

    def get_metadata_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get metadata for specific row indices (useful after FAISS search)."""
        filtered_table = self.table.take(indices)

        result = []
        for i in range(len(filtered_table)):
            row_data = {
                "id": filtered_table["id"][i].as_py(),
                "text": filtered_table["text"][i].as_py(),
                "label": filtered_table["label"][i].as_py(),
                "metadata": json.loads(filtered_table["metadata"][i].as_py()),
                "timestamp": filtered_table["timestamp"][i].as_py(),
            }
            result.append(row_data)

        return result

    def _table_to_dict(self, table: pa.Table) -> Dict[str, Any]:
        """Convert Arrow table to dictionary."""
        return {
            "ids": table["id"].to_pylist(),
            "doc_ids": table["doc_id"].to_pylist(),
            "texts": table["text"].to_pylist(),
            "embeddings": table["embedding"].to_pylist(),
            "labels": table["label"].to_pylist(),
            "metadata": [json.loads(m) for m in table["metadata"].to_pylist()],
            "timestamps": table["timestamp"].to_pylist(),
        }

    def save(self):
        """Save dataset to parquet file."""
        pq.write_table(self.table, self.parquet_path, compression="snappy")

        # Save metadata
        metadata = {
            "name": self.name,
            "embedding_dim": self.embedding_dim,
            "num_rows": len(self.table),
            "next_id": self._next_id,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return f"VectorDataset(name='{self.name}', rows={len(self.table)}, dim={self.embedding_dim})"

    def faiss_search(self, query_embedding, k=2) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(x=query_embedding, k=k)

        # Get metadata for results
        results = self.get_metadata_by_indices(indices[0])
        for i, result in enumerate(results):
            print(f"Result {i}: {result['text'][:50]}... (score: {distances[0][i]})")
        return results


if __name__ == "__main__":
    # Example usage and integration with FAISS
    def example_usage():
        """Demonstrate the vector dataset with FAISS integration."""

        # Create dataset
        dataset = VectorDataset("example_dataset", embedding_dim=128)

        # Add some sample data
        sample_embeddings = [
            np.random.rand(128).astype(np.float32),
            np.random.rand(128).astype(np.float32),
            np.random.rand(128).astype(np.float32),
        ]

        ids = dataset.add_batch(
            [
                {
                    "text": "This is a sample document about machine learning",
                    "embedding": sample_embeddings[0],
                    "label": "ML",
                    "metadata": {"category": "technical", "author": "user1"},
                },
                {
                    "text": "Another document about natural language processing",
                    "embedding": sample_embeddings[1],
                    "label": "NLP",
                    "metadata": {"category": "research", "author": "user2"},
                },
                {
                    "text": "A third document about computer vision",
                    "embedding": sample_embeddings[2],
                    "label": "CV",
                    "metadata": {"category": "technical", "author": "user1"},
                },
            ]
        )

        print(f"Added rows with IDs: {ids}")
        print(f"Dataset: {dataset}")

        # Get embeddings for FAISS (zero-copy)
        embeddings = dataset.get_embeddings_for_faiss()
        print(f"Embeddings shape for FAISS: {embeddings.shape}")

        # Query examples
        ml_docs = dataset.query_by_label("ML")
        print(f"ML documents: {len(ml_docs)}")

        tech_docs = dataset.query_by_text(".*technical.*")
        print(f"Technical documents: {len(tech_docs)}")

        # Save dataset
        dataset.save()
        print(f"Dataset saved to {dataset.parquet_path}")

        return dataset

    dataset = example_usage()
    dataset = VectorDataset("example_dataset")

    query_embedding = np.random.rand(1, dataset.embedding_dim).astype(np.float32)
    dataset.faiss_search(query_embedding)
