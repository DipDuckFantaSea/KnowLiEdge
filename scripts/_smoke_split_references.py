from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.chunking.md_chunker import load_markdown_doc, separate_main_text_and_references
from knotliedge.citation_graph.extract import extract_references_from_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: split references and extract citation edges.")
    parser.add_argument(
        "--md",
        type=str,
        required=True,
        help="Path to a markdown file (vault format with optional frontmatter).",
    )
    args = parser.parse_args()

    md_path = Path(args.md).resolve()
    doc = load_markdown_doc(md_path)

    main_text, references_text = separate_main_text_and_references(
        doc.body,
        reference_section_titles=[
            "References",
            "Reference",
            "Bibliography",
            "Works Cited",
            "Literature",
            "Literature Cited",
            "参考文献",
            "引用文献",
            "参考资料",
        ],
    )

    refs = extract_references_from_markdown(md_path.read_text(encoding="utf-8", errors="ignore"))

    print(f"doc_id={doc.doc_id} short_name={doc.short_name}")
    print(f"main_chars={len(main_text)} references_chars={len(references_text)} extracted_edges={len(refs)}")

    # Heuristic check: reference markers should not be in the main text tail if we split.
    tail = main_text[-1000:].lower()
    suspicious = any(x in tail for x in ["doi:", "https://doi.org", "arxiv:", "[1]", "1. "])
    print(f"tail_suspicious_reference_markers={suspicious}")


if __name__ == "__main__":
    main()

