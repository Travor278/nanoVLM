import multiprocessing
import unittest
from collections import Counter

from tests.worker_sharding_utils import (
    FakeStreamingDataset,
    LegacyVQADataset,
    NonShardableStreamingDataset,
    VQADataset,
    collect_ids_with_dataloader,
)


class TestWorkerSharding(unittest.TestCase):
    def assert_seen_once(self, seen_ids, sample_ids):
        sample_counts = Counter(seen_ids)

        self.assertEqual(
            len(seen_ids),
            len(sample_ids),
            f"Expected {len(sample_ids)} samples once each, saw {len(seen_ids)} items: {seen_ids}",
        )
        self.assertEqual(
            set(seen_ids),
            set(sample_ids),
            f"Expected all samples to be covered, saw {seen_ids}",
        )
        self.assertTrue(
            all(count == 1 for count in sample_counts.values()),
            f"Expected no duplicate samples, saw counts={dict(sample_counts)}",
        )

    def test_legacy_behavior_duplicates_samples_across_workers(self):
        sample_ids = list(range(2, 10))
        seen_ids = collect_ids_with_dataloader(
            LegacyVQADataset,
            FakeStreamingDataset(sample_ids),
            num_workers=2,
        )
        sample_counts = Counter(seen_ids)

        self.assertEqual(
            len(seen_ids),
            len(sample_ids) * 2,
            f"Expected each worker to read the full stream before the fix, saw {seen_ids}",
        )
        self.assertTrue(
            any(count > 1 for count in sample_counts.values()),
            f"Expected duplicate samples before the fix, saw counts={dict(sample_counts)}",
        )

    def test_shardable_streaming_dataset_is_not_duplicated_across_workers(self):
        sample_ids = list(range(2, 10))
        seen_ids = collect_ids_with_dataloader(
            VQADataset,
            FakeStreamingDataset(sample_ids),
            num_workers=2,
        )

        self.assert_seen_once(seen_ids, sample_ids)

    def test_non_shardable_streaming_dataset_falls_back_to_stride_sharding(self):
        sample_ids = list(range(2, 14))
        seen_ids = collect_ids_with_dataloader(
            VQADataset,
            NonShardableStreamingDataset(sample_ids),
            num_workers=3,
        )

        self.assert_seen_once(seen_ids, sample_ids)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    unittest.main()
