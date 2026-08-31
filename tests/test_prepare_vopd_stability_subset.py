import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_vopd_stability_subset import build_subset, select_row_indices


def make_table(count: int = 12) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "prompt": [{"role": "user", "content": f"question {index}"}],
                "images": [{"path": f"/full-{index}.png"}],
                "bbox_images": [{"path": f"/crop-{index}.png"}],
                "extra_info": {
                    "provenance": {
                        "sample_id": f"sample-{index:03d}",
                        "source_id": f"source-{index:03d}",
                        "source_row": index,
                        "group_id": f"group-{index:03d}",
                    }
                },
            }
            for index in range(count)
        ]
    )


class PrepareVopdStabilitySubsetTest(unittest.TestCase):
    def test_selection_is_independent_of_source_row_order(self):
        table = make_table()
        selected = select_row_indices(table, count=5, seed=42)
        reversed_table = table.take(pa.array(list(reversed(range(table.num_rows)))))
        reversed_selected = select_row_indices(reversed_table, count=5, seed=42)

        def ids(value, indices):
            rows = value.take(pa.array(indices)).to_pylist()
            return [row["extra_info"]["provenance"]["sample_id"] for row in rows]

        self.assertEqual(ids(table, selected), ids(reversed_table, reversed_selected))

    def test_builds_reproducible_subset_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.parquet"
            pq.write_table(make_table(), source)
            first_output = root / "first.parquet"
            second_output = root / "second.parquet"
            first = build_subset(source, first_output, root / "first.json", count=5, seed=42)
            second = build_subset(source, second_output, root / "second.json", count=5, seed=42)

            self.assertEqual(first["output"]["rows"], 5)
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
            self.assertEqual(
                [item["sample_id"] for item in first["samples"]],
                [item["sample_id"] for item in second["samples"]],
            )

    def test_rejects_duplicate_sample_ids(self):
        table = make_table(2)
        rows = table.to_pylist()
        rows[1]["extra_info"]["provenance"]["sample_id"] = "sample-000"
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            select_row_indices(pa.Table.from_pylist(rows), count=1, seed=42)


if __name__ == "__main__":
    unittest.main()
