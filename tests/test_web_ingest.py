import unittest
from scripts.web_ingest import parse_xhtml


SAMPLE = b"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Example Book | Standard Ebooks</title></head>
  <body>
    <section id="chapter-1" epub:type="chapter">
      <h2>Chapter I</h2>
      <p>First paragraph.</p>
      <p>Second paragraph.</p>
    </section>
    <section id="chapter-2" epub:type="chapter">
      <h2>Chapter II</h2>
      <p>Third paragraph.</p>
    </section>
  </body>
</html>
"""


class WebIngestTests(unittest.TestCase):
    def test_extracts_chapters_from_xhtml(self):
        _, chapters = parse_xhtml(SAMPLE)
        self.assertEqual(2, len(chapters))
        self.assertEqual("Chapter I", chapters[0][0])
        self.assertEqual("First paragraph.\n\nSecond paragraph.", chapters[0][1])
        self.assertEqual("Chapter II", chapters[1][0])


if __name__ == "__main__":
    unittest.main()
