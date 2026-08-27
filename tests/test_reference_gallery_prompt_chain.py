import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class ReferenceGalleryPromptChainTests(unittest.TestCase):
    def manifest(self, prompts):
        return json.dumps({
            "images": [{"file": "reference_gallery/subject.png", "name": "subject.png"}],
            "prompts": prompts,
        })

    def collect(self, prompts, **kwargs):
        return gallery.SECoursesReferenceGallery().collect(
            prompt="first @image1",
            references=self.manifest(prompts),
            video_fps=24.0,
            max_seconds=15.0,
            **kwargs,
        )

    def test_collects_added_prompts_as_one_shared_reference_batch(self):
        packs, prompts, active, merge, continuation, context = self.collect(["second", "third @image1"])

        self.assertEqual(prompts, ["first @image1", "second", "third @image1"])
        self.assertEqual(active, [True, True, True])
        self.assertEqual([pack["batch"]["index"] for pack in packs], [1, 2, 3])
        self.assertTrue(all(pack["batch"]["source"] == "prompt_chain" for pack in packs))
        self.assertTrue(all(pack["images"][0]["name"] == "subject.png" for pack in packs))
        self.assertEqual(merge, [False, False, False])
        self.assertEqual(continuation, [False, False, False])
        self.assertEqual(context, [1, 1, 1])

    def test_sequential_queue_selects_exactly_one_chained_prompt(self):
        packs, prompts, active, *_ = self.collect(
            ["second", "third"],
            batch_run_id="chain_12345678",
            batch_item_index=1,
            batch_item_count=3,
        )

        self.assertEqual(prompts, ["second"])
        self.assertEqual(active, [True])
        self.assertEqual(packs[0]["batch"]["run_id"], "chain_12345678")
        self.assertTrue(packs[0]["batch"]["sequential"])
        self.assertEqual(packs[0]["batch"]["index"], 2)

    def test_empty_added_prompt_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "Prompt 2.*empty"):
            gallery._parse_manifest(self.manifest(["  "]))

    def test_folder_batch_and_prompt_chain_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "one.txt").write_text("folder prompt", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "either Folder batch.*prompt chain"):
                self.collect(["second"], batch_folder=directory)


if __name__ == "__main__":
    unittest.main()
