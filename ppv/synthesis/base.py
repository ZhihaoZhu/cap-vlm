"""Base synthesizer for PPV data generation pipelines."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseSynthesizer(ABC):
    """Abstract base class for all PPV data synthesizers.

    Subclasses implement `_generate_single` to produce one sample from a source,
    and `_filter` to gate quality.  The `synthesize` method orchestrates parallel
    generation, filtering, and serialization.
    """

    def __init__(self, annotator_model: str, output_dir: str, num_workers: int = 4) -> None:
        self.annotator_model = annotator_model
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, sources: list[str], num_samples: int) -> None:
        """Generate up to `num_samples` from the given sources."""
        collected: list[dict] = []
        batch_idx = 0
        batch_size = 500

        if self.num_workers <= 1:
            iterator = map(self._generate_single, sources)
        else:
            pool = Pool(processes=self.num_workers)
            iterator = pool.imap_unordered(self._generate_single, sources)

        try:
            for sample in tqdm(iterator, total=len(sources), desc=self.__class__.__name__):
                if sample is None:
                    continue
                if not self._filter(sample):
                    continue
                collected.append(sample)

                if len(collected) >= batch_size:
                    self._save_batch(collected, batch_idx)
                    batch_idx += 1
                    collected = []

                if self._total_saved(batch_idx, len(collected)) >= num_samples:
                    break
        finally:
            if self.num_workers > 1:
                pool.terminate()
                pool.join()

        if collected:
            self._save_batch(collected, batch_idx)

        total = self._total_saved(batch_idx + (1 if collected else 0), 0)
        logger.info("Finished %s: %d samples written to %s", self.__class__.__name__, total, self.output_dir)

    @abstractmethod
    def _generate_single(self, source) -> dict | None:
        """Produce one training sample from *source*, or ``None`` on failure."""

    @abstractmethod
    def _filter(self, sample: dict) -> bool:
        """Return ``True`` if *sample* passes quality checks."""

    def _save_batch(self, samples: list[dict], batch_idx: int) -> None:
        """Persist a batch of samples as a JSONL shard."""
        path = self.output_dir / f"shard_{batch_idx:05d}.jsonl"
        with open(path, "w") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples to %s", len(samples), path)

    @staticmethod
    def _total_saved(completed_batches: int, current_buffer: int, batch_size: int = 500) -> int:
        return completed_batches * batch_size + current_buffer
