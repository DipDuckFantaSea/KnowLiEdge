from __future__ import annotations

from pathlib import Path

from knotliedge.topics.topic_discovery import run_topic_discovery


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Topic discovery (BERTopic) with BGE-M3 embeddings.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    args = parser.parse_args()
    run_topic_discovery(config_path=Path(args.config), limit=args.limit, min_cluster_size=int(args.min_cluster_size))


if __name__ == "__main__":
    main()
    raise SystemExit(0)

from pathlib import Path

from knotliedge.topics.topic_discovery import run_topic_discovery


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Topic discovery (BERTopic) with BGE-M3 embeddings.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    args = parser.parse_args()
    run_topic_discovery(config_path=Path(args.config), limit=args.limit, min_cluster_size=int(args.min_cluster_size))


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Optional

from knotliedge.topics.topic_discovery import (
    TopicDiscoveryArtifacts,
    interactive_show_intertopic_distance,
    interactive_train_topic_discovery,
    interactive_write_intertopic_distance_alias,
    run_topic_discovery,
    run_topic_discovery_with_artifacts,
)


def _build_custom_stopwords() -> List[str]:
    """Build merged stopwords for academic PDFs (English + academic filler + LaTeX residuals).

    Returns:
        A list of stopwords to pass into ``CountVectorizer``.
    """
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for topic_discovery stopwords. "
            "Please install `scikit-learn` in the `agent` env."
        ) from e

    academic_stopwords: Set[str] = {
        "et",
        "al",
        "figure",
        "table",
        "method",
        "proposed",
        "results",
        "using",
        "study",
        "analysis",
        "paper",
        "based",
        "fig",
        "eq",
        "equation",
        "show",
        "data",
        "model",
    }
    latex_stopwords: Set[str] = {
        "mathrm",
        "mathbf",
        "frac",
        "partial",
        "mu",
        "varepsilon",
        "omega",
        "alpha",
        "beta",
        "gamma",
        "sum",
        "int",
        "approx",
        "left",
        "right",
        "text",
        "textit",
        "textbf",
    }

    # Template/layout residuals (best-effort): these appear frequently in MinerU/HTML-ish
    # conversions and can dominate topics if not filtered.
    template_stopwords: Set[str] = {
        # table / html
        "row",
        "rows",
        "column",
        "columns",
        "cell",
        "cells",
        "rowspan",
        "colspan",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "nbsp",
        "html",
        "href",
        # publisher / venues
        "ieee",
        "arxiv",
        "preprint",
        "copyright",
        "license",
        # citations / structure markers
        "doi",
        "vol",
        "no",
        "pp",
        "appendix",
        "section",
        "sec",
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgment",
        "acknowledgements",
        # common conversion artifacts
        "prime",
        "line",
        "lines",
        "page",
        "pages",
    }

    merged = (
        set(ENGLISH_STOP_WORDS)
        .union(academic_stopwords)
        .union(latex_stopwords)
        .union(template_stopwords)
    )
    return sorted({str(w).strip().lower() for w in merged if str(w).strip()})


_RE_BACKSLASH_CMD = re.compile(r"\\[A-Za-z]+")
_RE_ISOLATED_NUMBER = re.compile(r"\b\d+\b")
_RE_SINGLE_LETTER = re.compile(r"\b[a-zA-Z]\b")
_RE_ALPHA_NUM_GLUE = re.compile(r"\b([a-zA-Z]+)(\d+)\b")
_RE_MATH_SYMBOLS = re.compile(r"[_^{}=<>~|*/+\-]+")
_RE_NONWORD = re.compile(r"[^a-zA-Z0-9\s]+")
_RE_WS = re.compile(r"\s+")


def _clean_text_for_vectorizer(text: str) -> str:
    """Lightweight cleaning for CountVectorizer to avoid LaTeX/formatting dominating topics.

    Steps:
    - remove backslash commands like ``\\mathrm`` (leave arguments to be tokenized)
    - replace backslashes and math-ish symbols with spaces (avoid glueing to tokens)
    - drop isolated numbers and single-letter tokens
    - normalize whitespace

    Args:
        text: Raw document string.

    Returns:
        Cleaned text.
    """
    s = str(text or "")
    if not s:
        return ""
    s = _RE_BACKSLASH_CMD.sub(" ", s)
    s = s.replace("\\", " ")
    s = _RE_MATH_SYMBOLS.sub(" ", s)
    s = _RE_NONWORD.sub(" ", s)
    s = s.lower()
    # Split glued patterns like "rowspan1" -> "rowspan 1" so stopwords can match.
    s = _RE_ALPHA_NUM_GLUE.sub(r"\1 \2", s)
    s = _RE_ISOLATED_NUMBER.sub(" ", s)
    s = _RE_SINGLE_LETTER.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


def _build_vectorizer_model(*, min_df: int = 2) -> object:
    """Create the CountVectorizer used by BERTopic for c-TF-IDF features.

    Args:
        min_df: Filter out terms appearing in fewer than min_df documents.

    Returns:
        A configured CountVectorizer instance.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "CountVectorizer requires scikit-learn. Please install `scikit-learn` in the `agent` env."
        ) from e

    custom_stop_words = _build_custom_stopwords()
    return CountVectorizer(
        stop_words=custom_stop_words,
        ngram_range=(1, 2),
        min_df=int(min_df),
        preprocessor=_clean_text_for_vectorizer,
    )


@dataclass(frozen=True)
class DocItem:
    doc_id: str
    short_name: str
    title: str
    source_md: str
    text: str


def _iter_vault_docs(vault_dir: Path, *, limit: Optional[int] = None) -> List[DocItem]:
    md_files = sorted(vault_dir.rglob("*.md"))
    if limit is not None:
        md_files = md_files[: int(limit)]
    out: List[DocItem] = []
    for p in md_files:
        try:
            doc = load_markdown_doc(p)
            out.append(
                DocItem(
                    doc_id=str(doc.doc_id),
                    short_name=str(doc.short_name),
                    title=str(doc.title),
                    source_md=str(p.resolve()),
                    text=str(doc.body or ""),
                )
            )
        except Exception as e:
            logger.warning("Failed to read markdown: %s | %s", p, e)
    return out


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_write_html(fig: object, out_path: Path) -> None:
    try:
        # plotly figures have write_html
        getattr(fig, "write_html")(str(out_path))
    except Exception as e:
        logger.warning("Failed to write html: %s | %s", out_path, e)


def _escape_html(text: str) -> str:
    s = text or ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _plotly_fig_to_div(fig: object, *, include_plotlyjs: Union[str, bool], div_id: str) -> str:
    """Convert a Plotly figure to an HTML div fragment.

    Args:
        fig: Plotly figure-like object with ``to_html``.
        include_plotlyjs: Plotly include_plotlyjs flag (``True``/``False``/``"cdn"``).
        div_id: Stable div id for anchoring.

    Returns:
        HTML fragment string.
    """
    to_html = getattr(fig, "to_html")
    return str(
        to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            div_id=str(div_id),
        )
    )


def _write_multi_plotly_report_html(
    *,
    out_path: Path,
    title: str,
    sections: List[Tuple[str, str, object]],
) -> None:
    """Write a single self-contained HTML bundling multiple Plotly figures.

    This is additive: it does not replace per-figure exports.

    Args:
        out_path: Destination HTML path.
        title: Page title.
        sections: List of (anchor_id, heading, fig).

    Returns:
        None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nav_items: List[str] = []
    body_sections: List[str] = []
    first_plot_written = False
    for anchor, heading, fig in sections:
        if fig is None:
            continue
        nav_items.append(f'<a href="#{anchor}">{_escape_html(heading)}</a>')
        include_js: Union[str, bool] = "cdn" if not first_plot_written else False
        div_html = _plotly_fig_to_div(fig, include_plotlyjs=include_js, div_id=anchor)
        first_plot_written = True
        body_sections.append(
            f"""
<section class="section" id="{_escape_html(anchor)}">
  <h2>{_escape_html(heading)}</h2>
  {div_html}
</section>
"""
        )

    nav_html = " | ".join(nav_items) if nav_items else "<i>No figures available.</i>"

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape_html(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; margin: 0; background: #0b1020; color: #e7e9ee; }}
    header {{ position: sticky; top: 0; z-index: 10; backdrop-filter: blur(10px); background: rgba(11, 16, 32, 0.85); border-bottom: 1px solid rgba(255,255,255,0.08); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    h1 {{ font-size: 18px; margin: 0 0 8px 0; letter-spacing: 0.2px; }}
    nav a {{ color: #9bd3ff; text-decoration: none; margin-right: 10px; }}
    nav a:hover {{ text-decoration: underline; }}
    .section {{ margin-top: 22px; padding: 14px; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; background: rgba(255,255,255,0.03); }}
    h2 {{ font-size: 15px; margin: 0 0 10px 0; color: #cfe6ff; }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{_escape_html(title)}</h1>
      <nav>{nav_html}</nav>
    </div>
  </header>
  <main class="wrap">
    {"".join(body_sections)}
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _maybe_show_plotly(fig: object, *, label: str) -> None:
    """Best-effort interactive display for Cursor Interactive Window.

    Args:
        fig: Plotly figure-like object.
        label: Label for logging.

    Returns:
        None.
    """
    try:
        show = getattr(fig, "show")
        show()
    except Exception as e:
        logger.info("Plotly show skipped (%s): %s", label, e)


def _maybe_rich_topic_table(*, topic_model: object, top_n: int = 15) -> None:
    """Render a compact topic table using Rich (additive UX).

    Args:
        topic_model: BERTopic model instance.
        top_n: Max rows.

    Returns:
        None.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        return

    try:
        info = topic_model.get_topic_info()
        rows = info.head(int(top_n)).to_dict(orient="records") if hasattr(info, "head") else []
    except Exception:
        return

    console = Console(stderr=True, force_terminal=True, legacy_windows=True, emoji=False, highlight=False, markup=False)
    table = Table(title=f"BERTopic topic_info (top {int(top_n)})", show_lines=False)
    table.add_column("Topic", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Name", overflow="ellipsis", max_width=90)

    for r in rows:
        topic = r.get("Topic")
        count = r.get("Count")
        name = str(r.get("Name") or "")
        table.add_row(str(topic), str(count), name)

    console.print(table)


def _count_assigned_topics(topics: List[int]) -> int:
    """Count topics excluding the outlier topic (-1).

    Args:
        topics: Topic id per document.

    Returns:
        Number of unique assigned topics.
    """
    s = {int(t) for t in topics}
    s.discard(-1)
    return int(len(s))


def _maybe_refit_bertopic_for_visualization(
    *,
    texts: List[str],
    embeddings_np: "np.ndarray",
    umap_metric: str,
    umap_n_neighbors: int,
    umap_n_components: int,
    min_cluster_size: int,
    n_docs: int,
    first_topic_model: object,
    first_topics: List[int],
    first_probs: object,
    vectorizer_model: Optional[object],
) -> Tuple[object, List[int], object]:
    """Refit BERTopic with milder clustering if the first fit yields almost only outliers.

    This is primarily for **local validation** on small/medium corpora where HDBSCAN may
    label everything as noise (-1), which breaks several Plotly visualizations.

    Args:
        texts: Document texts.
        embeddings_np: Precomputed embeddings (float32 ndarray).
        umap_metric: UMAP metric (should remain cosine for BGE-M3).
        umap_n_neighbors: Requested UMAP n_neighbors (may be clipped earlier).
        umap_n_components: Requested UMAP n_components (may be clipped earlier).
        min_cluster_size: Requested HDBSCAN min_cluster_size (may be clipped earlier).
        n_docs: Number of documents.
        first_topic_model: First fitted BERTopic model.
        first_topics: Topics from first fit.
        first_probs: Probabilities from first fit.

    Returns:
        Tuple of (topic_model, topics, probs).
    """
    try:
        from bertopic import BERTopic  # type: ignore
        import umap  # type: ignore
        import hdbscan  # type: ignore
    except Exception:
        return first_topic_model, first_topics, first_probs

    assigned = _count_assigned_topics([int(t) for t in first_topics])
    outlier_ratio = float(sum(1 for t in first_topics if int(t) == -1)) / float(max(1, n_docs))
    if assigned >= 2 and outlier_ratio < 0.95:
        return first_topic_model, first_topics, first_probs

    logger.warning(
        "BERTopic produced few assigned topics (assigned_unique=%s, outlier_ratio=%.3f). Refitting with milder HDBSCAN for visualization stability.",
        assigned,
        outlier_ratio,
    )

    max_neighbors = max(2, n_docs - 1)
    eff_neighbors = max(2, min(int(umap_n_neighbors), max_neighbors))
    # UMAP spectral init may fail for very small N when k >= N. Keep components < N.
    # Empirically, clipping to (N-2) avoids scipy edge cases for N≈4..10.
    max_components = max(1, n_docs - 2)
    eff_components = max(2, min(int(umap_n_components), max_components))
    if n_docs <= 3:
        eff_components = 1
        eff_neighbors = min(eff_neighbors, max(2, n_docs - 1))

    # Milder clustering: smaller min_cluster_size + slightly smaller min_samples.
    mild_min_cluster = max(2, min(int(min_cluster_size), max(2, n_docs // 3)))
    mild_min_samples = max(1, mild_min_cluster // 2)

    umap_model = umap.UMAP(
        n_neighbors=int(eff_neighbors),
        n_components=int(eff_components),
        metric=str(umap_metric),
        random_state=42,
        init="random" if n_docs <= 10 else "spectral",
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(mild_min_cluster),
        min_samples=int(mild_min_samples),
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings_np)
    return topic_model, [int(t) for t in topics], probs


def _bertopic_non_outlier_topic_ids(*, topic_model: object) -> List[int]:
    """Return sorted non-outlier topic ids from BERTopic's topic info table.

    Args:
        topic_model: A fitted ``BERTopic`` instance.

    Returns:
        Topic ids excluding ``-1`` (HDBSCAN noise), sorted ascending.
    """
    info = topic_model.get_topic_info()
    if info is None:
        return []
    topic_col = "Topic" if "Topic" in getattr(info, "columns", []) else None
    if topic_col is None:
        return []
    ids: List[int] = []
    for raw in list(info[topic_col].tolist()):
        try:
            tid = int(raw)
        except Exception:
            continue
        if tid == -1:
            continue
        ids.append(tid)
    return sorted(set(ids))


def _pca_xy_two_columns(*, emb: np.ndarray) -> np.ndarray:
    """Reduce rows of ``emb`` to 2 columns for scatter (PCA or SVD fallback).

    Args:
        emb: 2D array, shape ``(n_topics, dim)``.

    Returns:
        Array of shape ``(n_topics, 2)`` suitable for x/y scatter coordinates.
    """
    x = np.asarray(emb, dtype=np.float64)
    if x.ndim != 2 or int(x.shape[0]) < 2:
        raise ValueError(f"emb must be 2D with at least 2 rows, got shape={getattr(x, 'shape', None)}")
    n, d = int(x.shape[0]), int(x.shape[1])
    try:
        from sklearn.decomposition import PCA  # type: ignore

        k = int(min(2, n, d))
        if k < 1:
            raise ValueError("no PCA components")
        xy = PCA(n_components=k, random_state=42).fit_transform(x)
        if xy.shape[1] == 1:
            return np.column_stack([xy[:, 0], np.zeros(n, dtype=np.float64)])
        return np.asarray(xy[:, :2], dtype=np.float64)
    except Exception:
        x0 = x - np.mean(x, axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(x0, full_matrices=False)
        c1 = u[:, 0] * float(s[0]) if s.size else u[:, 0]
        if n > 1 and s.size > 1 and float(s[1]) > 1e-12:
            c2 = u[:, 1] * float(s[1])
        else:
            c2 = np.linspace(-0.05, 0.05, n, dtype=np.float64)
        return np.column_stack([c1, c2])


def _plotly_intertopic_scatter_fallback(*, topic_model: object) -> Optional[object]:
    """2D scatter of topic embeddings when BERTopic's ``visualize_topics`` fails.

    Uses the same representation selection as BERTopic plotting (topic embeddings vs
    c-TF-IDF), then PCA (or SVD) to two axes. Title indicates this is a fallback, not
    the official intertopic distance map.

    Args:
        topic_model: A fitted ``BERTopic`` instance.

    Returns:
        A Plotly ``Figure``, or ``None`` if insufficient data.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
        from bertopic._utils import select_topic_representation  # type: ignore
    except Exception as e:
        logger.warning("intertopic PCA fallback: import failed: %s", e)
        return None

    try:
        freq_df = topic_model.get_topic_freq()
    except Exception as e:
        logger.warning("intertopic PCA fallback: get_topic_freq failed: %s", e)
        return None
    if freq_df is None or not len(freq_df):
        return None
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if len(freq_df) < 2:
        logger.warning("intertopic PCA fallback: need at least 2 non-outlier topics, got %s", len(freq_df))
        return None

    topic_list = [int(t) for t in sorted(freq_df.Topic.tolist())]
    try:
        all_topics = sorted(list(topic_model.get_topics().keys()))
        indices = np.array([all_topics.index(int(t)) for t in topic_list], dtype=np.intp)
        embeddings, _used = select_topic_representation(
            topic_model.c_tf_idf_,
            topic_model.topic_embeddings_,
            use_ctfidf=False,
            output_ndarray=True,
        )
        row_emb = np.asarray(embeddings[indices], dtype=np.float64)
    except Exception as e:
        logger.warning("intertopic PCA fallback: embedding extraction failed: %s", e)
        return None

    if row_emb.ndim != 2 or int(row_emb.shape[0]) < 2:
        logger.warning("intertopic PCA fallback: bad embedding shape %s", getattr(row_emb, "shape", None))
        return None

    try:
        xy = _pca_xy_two_columns(emb=row_emb)
    except Exception as e:
        logger.warning("intertopic PCA fallback: dimension reduction failed: %s", e)
        return None

    sizes_src = getattr(topic_model, "topic_sizes_", None)
    frequencies: List[int] = []
    for tid in topic_list:
        try:
            if sizes_src is not None and int(tid) in sizes_src:
                frequencies.append(int(sizes_src[int(tid)]))
            else:
                row = freq_df.loc[freq_df.Topic == int(tid), "Count"]
                frequencies.append(int(row.iloc[0]) if len(row) else 1)
        except Exception:
            frequencies.append(1)

    words: List[str] = []
    for tid in topic_list:
        try:
            top = topic_model.get_topic(int(tid))
            words.append(" | ".join([str(w[0]) for w in top[:5]]))
        except Exception:
            words.append(str(tid))

    sizes = [int(max(12, min(64, 4.0 * np.sqrt(float(c))))) for c in frequencies]

    title = (
        "<b>Intertopic map (PCA fallback)</b><br>"
        "<sup>BERTopic visualize_topics failed; 2D projection of topic-level representation</sup>"
    )
    hover_lines = [
        f"<b>Topic {tid}</b><br>count={cnt}<br>{_escape_html(w[:200])}"
        for tid, cnt, w in zip(topic_list, frequencies, words)
    ]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xy[:, 0].tolist(),
                y=xy[:, 1].tolist(),
                mode="markers",
                marker=dict(size=sizes, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")),
                text=[f"T{t}" for t in topic_list],
                hovertext=hover_lines,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=680,
        width=720,
    )
    logger.info("intertopic PCA fallback figure built (%s topics)", len(topic_list))
    return fig


def _ensure_topic_embeddings_for_visualize(
    *, topic_model: object, embeddings_np: "np.ndarray", topics: Sequence[int]
) -> None:
    """Best-effort ensure ``topic_model.topic_embeddings_`` is present for visualize_topics().

    Some BERTopic versions can end up with missing/empty topic embeddings on small corpora,
    leading to failures inside ``visualize_topics``. Since we already have per-doc embeddings,
    we can construct topic centroids and inject them back into the model.

    Args:
        topic_model: Fitted BERTopic model.
        embeddings_np: Per-document embeddings used for fitting, shape (n_docs, dim).
        topics: Topic id per document.

    Returns:
        None.
    """
    try:
        all_topics = sorted(list(topic_model.get_topics().keys()))
    except Exception:
        return

    try:
        cur = getattr(topic_model, "topic_embeddings_", None)
        if cur is not None:
            arr = np.asarray(cur)
            if arr.ndim == 2 and int(arr.shape[0]) == int(len(all_topics)) and int(arr.shape[1]) >= 2:
                return
    except Exception:
        pass

    try:
        emb = np.asarray(embeddings_np, dtype=np.float32)
        if emb.ndim != 2 or int(emb.shape[0]) != int(len(list(topics))):
            return
        dim = int(emb.shape[1])
        topic_arr = np.asarray(list(topics), dtype=np.int64)
        out = np.zeros((len(all_topics), dim), dtype=np.float32)
        for i, tid in enumerate(all_topics):
            if int(tid) == -1:
                continue
            mask = topic_arr == int(tid)
            if not np.any(mask):
                continue
            out[i, :] = np.mean(emb[mask, :], axis=0)
        setattr(topic_model, "topic_embeddings_", out)
        logger.info("Injected topic_embeddings_ for visualize_topics (n_topics=%s dim=%s)", len(all_topics), dim)
    except Exception as e:
        logger.warning("Failed to inject topic_embeddings_ for visualize_topics: %s", e)


def _visualize_topics_with_fallback(
    *, topic_model: object, embeddings_np: "np.ndarray", topics: Sequence[int]
) -> Optional[object]:
    """Create the BERTopic intertopic distance figure with small-corpus fallbacks.

    Args:
        topic_model: A fitted ``BERTopic`` instance.
        embeddings_np: Per-document embeddings used for fitting.
        topics: Topic id per document.

    Returns:
        A Plotly figure-like object, or ``None`` if visualization is not possible.
    """
    topic_ids = _bertopic_non_outlier_topic_ids(topic_model=topic_model)
    # With too few topics, the official visualize_topics is both less meaningful and more likely
    # to hit edge cases internally. Prefer a stable fallback in that case.
    min_topics_for_official = 5
    if len(topic_ids) < 2:
        logger.warning("No intertopic figure: not enough non-outlier topics (n=%s).", len(topic_ids))
        return None
    if len(topic_ids) < int(min_topics_for_official):
        logger.info(
            "Skip official visualize_topics: too few non-outlier topics (n=%s < %s). Using PCA fallback.",
            len(topic_ids),
            int(min_topics_for_official),
        )
        return _plotly_intertopic_scatter_fallback(topic_model=topic_model)

    _ensure_topic_embeddings_for_visualize(topic_model=topic_model, embeddings_np=embeddings_np, topics=topics)
    try:
        return topic_model.visualize_topics(topics=topic_ids)
    except Exception as e:
        logger.warning("visualize_topics failed: %s", e)
        # Re-ensure embeddings and retry once (some versions populate lazily after failures)
        _ensure_topic_embeddings_for_visualize(topic_model=topic_model, embeddings_np=embeddings_np, topics=topics)
        try:
            return topic_model.visualize_topics(topics=topic_ids)
        except Exception as e2:
            logger.warning("visualize_topics (retry) failed: %s", e2)

    fig_fb = _plotly_intertopic_scatter_fallback(topic_model=topic_model)
    if fig_fb is not None:
        return fig_fb
    return None


@dataclass(frozen=True)
class TopicDiscoveryArtifacts:
    """Artifacts produced by topic discovery (for Interactive Window workflows)."""

    out_dir: Path
    topic_model: object
    topics: List[int]
    probs: object
    fig_topics: Optional[object]
    fig_barchart: Optional[object]
    fig_hierarchy: Optional[object]
    fig_heatmap: Optional[object]


def run_topic_discovery_with_artifacts(
    *,
    config_path: Path,
    limit: Optional[int] = None,
    min_cluster_size: int = 5,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_metric: str = "cosine",
    out_dir: Optional[Path] = None,
) -> TopicDiscoveryArtifacts:
    """Run BERTopic topic discovery and return model + figures for interactive use.

    Args:
        config_path: YAML config path.
        limit: Optional max number of markdown docs to use.
        min_cluster_size: HDBSCAN min cluster size.
        umap_n_neighbors: UMAP n_neighbors.
        umap_n_components: UMAP n_components.
        umap_metric: UMAP metric. For BGE-M3 this MUST be 'cosine'.
        out_dir: Optional output directory root. Default: `output/topics/<timestamp>/`.

    Returns:
        TopicDiscoveryArtifacts.
    """
    cfg = load_app_config(Path(config_path))
    items = _iter_vault_docs(cfg.paths.markdown_vault_dir, limit=limit)
    if not items:
        raise RuntimeError(f"No markdown files found under: {cfg.paths.markdown_vault_dir}")

    texts = [it.text for it in items]
    n_docs = int(len(texts))

    embedder = get_embedder(config_path=Path(config_path))
    embeddings = embedder.embed_texts(texts)
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings_np.shape}")
    if int(embeddings_np.shape[0]) != n_docs:
        raise ValueError(f"Embeddings/doc mismatch: n_docs={n_docs} emb_rows={int(embeddings_np.shape[0])}")

    try:
        from bertopic import BERTopic  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "BERTopic is not installed. Please install optional deps: bertopic, umap-learn, hdbscan, plotly."
        ) from e

    try:
        import umap  # type: ignore
    except Exception as e:
        raise RuntimeError("umap-learn is not installed. Please install optional deps: umap-learn.") from e

    try:
        import hdbscan  # type: ignore
    except Exception as e:
        raise RuntimeError("hdbscan is not installed. Please install optional deps: hdbscan.") from e

    # UMAP/HDBSCAN hyper-params must remain valid for very small corpora (e.g. smoke tests with --limit 3).
    # UMAP requires: 2 <= n_neighbors <= n_samples - 1
    max_neighbors = max(2, n_docs - 1)
    eff_neighbors = max(2, min(int(umap_n_neighbors), max_neighbors))
    # Keep components < N to avoid UMAP spectral init scipy edge cases.
    max_components = max(1, n_docs - 2)
    eff_components = max(2, min(int(umap_n_components), max_components))
    eff_min_cluster_size = max(2, min(int(min_cluster_size), n_docs))

    # For extremely small corpora, UMAP's spectral init can hit scipy edge cases when n_components>1.
    # Keep smoke tests reliable by forcing 1D embedding for N<=3.
    if n_docs <= 3:
        eff_components = 1
        eff_neighbors = min(eff_neighbors, max(2, n_docs - 1))

    if eff_neighbors != int(umap_n_neighbors) or eff_components != int(umap_n_components) or eff_min_cluster_size != int(min_cluster_size):
        logger.warning(
            "Adjusted clustering hyperparams for small corpus: n_docs=%s umap_n_neighbors %s->%s n_components %s->%s min_cluster_size %s->%s",
            n_docs,
            int(umap_n_neighbors),
            eff_neighbors,
            int(umap_n_components),
            eff_components,
            int(min_cluster_size),
            eff_min_cluster_size,
        )

    # Explicit models per tuning strategy (BGE-M3 + mid-size local corpus)
    vectorizer_model = _build_vectorizer_model(min_df=2)
    umap_model = umap.UMAP(
        n_neighbors=int(eff_neighbors),
        n_components=int(eff_components),
        metric=str(umap_metric),
        random_state=42,
        init="random" if n_docs <= 10 else "spectral",
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(eff_min_cluster_size),
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings_np)
    topics_i = [int(t) for t in topics]

    topic_model, topics_i, probs = _maybe_refit_bertopic_for_visualization(
        texts=texts,
        embeddings_np=embeddings_np,
        umap_metric=str(umap_metric),
        umap_n_neighbors=int(umap_n_neighbors),
        umap_n_components=int(umap_n_components),
        min_cluster_size=int(min_cluster_size),
        n_docs=n_docs,
        first_topic_model=topic_model,
        first_topics=topics_i,
        first_probs=probs,
        vectorizer_model=vectorizer_model,
    )

    _maybe_rich_topic_table(topic_model=topic_model, top_n=15)

    ts = _now_ts()
    out_root = Path(out_dir) if out_dir is not None else (cfg.project_root / "output" / "topics" / ts)
    out_root = out_root.resolve()
    viz_dir = out_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    info = topic_model.get_topic_info()
    topics_payload: Dict[str, object] = {
        "config_path": str(Path(config_path).resolve()),
        "created_at": ts,
        "n_docs": len(items),
        "umap_requested": {
            "metric": str(umap_metric),
            "n_neighbors": int(umap_n_neighbors),
            "n_components": int(umap_n_components),
        },
        "umap_effective": {
            "metric": str(umap_metric),
            "n_neighbors": int(eff_neighbors),
            "n_components": int(eff_components),
        },
        "hdbscan_requested": {"min_cluster_size": int(min_cluster_size)},
        "hdbscan_effective": {"min_cluster_size": int(eff_min_cluster_size)},
        "topic_info": info.to_dict(orient="records") if hasattr(info, "to_dict") else [],
        "docs": [
            {
                "doc_id": it.doc_id,
                "short_name": it.short_name,
                "title": it.title,
                "source_md": it.source_md,
                "topic": int(t),
            }
            for it, t in zip(items, topics_i)
        ],
    }
    _write_json(out_root / "topics.json", topics_payload)

    fig_topics = None
    fig_barchart = None
    fig_hierarchy = None
    fig_heatmap = None

    fig_topics = _visualize_topics_with_fallback(topic_model=topic_model, embeddings_np=embeddings_np, topics=topics_i)
    if fig_topics is not None:
        _try_write_html(fig_topics, viz_dir / "topics.html")
    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=15)
        _try_write_html(fig_barchart, viz_dir / "barchart.html")
    except Exception as e:
        logger.warning("visualize_barchart failed: %s", e)
    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        _try_write_html(fig_hierarchy, viz_dir / "hierarchy.html")
    except Exception as e:
        logger.warning("visualize_hierarchy failed: %s", e)
    try:
        fig_heatmap = topic_model.visualize_heatmap()
        _try_write_html(fig_heatmap, viz_dir / "heatmap.html")
    except Exception as e:
        logger.warning("visualize_heatmap failed: %s", e)

    try:
        report_sections: List[Tuple[str, str, object]] = [
            ("sec_topics", "Intertopic distance map", fig_topics),
            ("sec_barchart", "Topic term barchart", fig_barchart),
            ("sec_hierarchy", "Topic hierarchy", fig_hierarchy),
            ("sec_heatmap", "Topic similarity heatmap", fig_heatmap),
        ]
        _write_multi_plotly_report_html(
            out_path=viz_dir / "topic_report.html",
            title="KnotLiEdge / BERTopic visualization report",
            sections=report_sections,
        )
        logger.info("Bundled HTML report written: %s", viz_dir / "topic_report.html")
    except Exception as e:
        logger.warning("Failed to write bundled HTML report: %s", e)

    logger.info("Topic discovery done. out_dir=%s", out_root)
    return TopicDiscoveryArtifacts(
        out_dir=out_root,
        topic_model=topic_model,
        topics=topics_i,
        probs=probs,
        fig_topics=fig_topics,
        fig_barchart=fig_barchart,
        fig_hierarchy=fig_hierarchy,
        fig_heatmap=fig_heatmap,
    )


# %%
# 交互入口：在 Cursor Interactive Window 中运行本 cell（CLI 也复用该实现）
def run_topic_discovery(
    *,
    config_path: Path,
    limit: Optional[int] = None,
    min_cluster_size: int = 5,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_metric: str = "cosine",
    out_dir: Optional[Path] = None,
) -> Path:
    """Run BERTopic topic discovery with explicit UMAP + HDBSCAN configs.

    Notes:
        - We inject precomputed BGE-M3 embeddings (1024 dims) from our embedder.
        - We explicitly define UMAP/HDBSCAN to avoid "curse of dimensionality".

    Args:
        config_path: YAML config path.
        limit: Optional max number of markdown docs to use.
        min_cluster_size: HDBSCAN min cluster size.
        umap_n_neighbors: UMAP n_neighbors.
        umap_n_components: UMAP n_components.
        umap_metric: UMAP metric. For BGE-M3 this MUST be 'cosine'.
        out_dir: Optional output directory root. Default: `output/topics/<timestamp>/`.

    Returns:
        Output directory path containing topics.json and viz html files.
    """
    art = run_topic_discovery_with_artifacts(
        config_path=config_path,
        limit=limit,
        min_cluster_size=min_cluster_size,
        umap_n_neighbors=umap_n_neighbors,
        umap_n_components=umap_n_components,
        umap_metric=umap_metric,
        out_dir=out_dir,
    )
    return art.out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Topic discovery (BERTopic) with BGE-M3 embeddings.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    args = parser.parse_args()
    run_topic_discovery(config_path=Path(args.config), limit=args.limit, min_cluster_size=int(args.min_cluster_size))


if __name__ == "__main__":
    main()
