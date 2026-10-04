import unittest

from scripts.wb_ingest import (
    assert_storage_policy,
    packetize_paragraphs,
    profile,
    word_count,
    build_public_handoff,
)


class WBIngestTests(unittest.TestCase):
    def test_profile_uses_wb01_break_rules(self):
        text = (
            'A long narration sentence continues here for several words. '
            'Short one. Another clause; final clause.\n\n'
            '“Hello there,” she said. Narration returns: briefly.\n\n'
            'One—two.'
        )
        got = profile(text)
        self.assertGreater(got["words"], 0)
        self.assertGreater(got["narration_sentences"], 0)
        self.assertGreaterEqual(got["short_sent"], 0)
        self.assertGreater(got["dialogue_share"], 0)
        self.assertGreater(got["semicolons_1k"], 0)
        self.assertGreater(got["emdash_1k"], 0)

    def test_public_domain_may_be_tracked(self):
        assert_storage_policy("public_domain", "tracked")
        assert_storage_policy("public_domain_us", "tracked")

    def test_owned_cannot_be_tracked(self):
        with self.assertRaises(ValueError):
            assert_storage_policy("owned", "tracked")
        with self.assertRaises(ValueError):
            assert_storage_policy("excerpt", "tracked")

    def test_packetizer_respects_normal_bounds(self):
        paras = [("word " * 120).strip() for _ in range(20)]
        packets = packetize_paragraphs(paras, min_words=700, target_words=1000, max_words=1500)
        self.assertGreater(len(packets), 1)
        counts = [sum(word_count(p) for p in packet) for packet in packets]
        for count in counts[:-1]:
            self.assertLessEqual(count, 1500)
            self.assertGreaterEqual(count, 700)

    def test_handoff_authority_is_single_target(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "library" / "author" / "book"
            chapters = book / "chapters"
            chapters.mkdir(parents=True)
            (book / "metadata.json").write_text(json.dumps({
                "title": "Book",
                "author": "Author",
                "source_type": "standard_ebooks",
                "rights": "public_domain_us",
                "output_slug": "author/book"
            }), encoding="utf-8")
            (chapters / "001-one.md").write_text("# One\n\n" + ("word " * 800), encoding="utf-8")
            out = build_public_handoff(book, root / "wb")
            handoff = json.loads((out / "handoff.json").read_text(encoding="utf-8"))
            authority = handoff["authority"]
            self.assertTrue(authority["writing_bloom_target"])
            self.assertFalse(authority["project_bloom_write"])
            self.assertFalse(authority["personal_global_bloom_write"])
            self.assertFalse(authority["other_domain_bloom_write"])
            self.assertFalse(authority["notion_write"])


if __name__ == "__main__":
    unittest.main()
