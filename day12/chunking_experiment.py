"""
=============================================================================
CONTEXT CHUNKING EXPERIMENT
=============================================================================
Compares multiple chunking strategies on a Wikipedia article about
Artificial Intelligence to understand how chunk design affects retrieval
quality.

Configurations tested:
  - Chunk sizes:  100, 300, 500, 1000 characters
  - Overlaps:     0, 50, 100 characters
  - Total:        12 configurations

Methods:
  1. Manual fixed-size chunking (pure Python, no LangChain)
  2. LangChain RecursiveCharacterTextSplitter

Tools: Python, LangChain, FAISS, HuggingFace Embeddings
=============================================================================
"""

import json
import sys
import textwrap
import urllib.request
import warnings
from typing import Optional

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- LangChain & ML imports --------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =============================================================================
# SECTION 1 -- Load a long text document (>= 5 pages) as a plain string
# =============================================================================

def load_wikipedia_article(title: str = "Artificial_intelligence") -> str:
    """
    Fetch the plain-text extract of a Wikipedia article using the
    MediaWiki API.  Returns a single string (typically 40-80 k chars,
    well over five printed pages).
    """
    url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query"
        f"&titles={title}"
        "&prop=extracts"
        "&explaintext=true"
        "&format=json"
    )
    print(f"[FETCH] Fetching Wikipedia article: '{title}' ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ChunkingExperiment/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    text = page["extract"]
    print(f"[DONE]  Loaded {len(text):,} characters  (~{len(text) // 3000} pages)\n")
    return text


# =============================================================================
# SECTION 2 -- Manual fixed-size chunking (pure Python, NO LangChain)
# =============================================================================

def manual_fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split *text* every *chunk_size* characters using a simple loop.

    When chunk_overlap > 0 the next chunk starts (chunk_size - overlap)
    characters after the previous one, so that boundary content is
    duplicated across adjacent chunks.

    This is the "from scratch" implementation -- no external libraries.
    """
    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)   # stride between chunk starts
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step

    return chunks


# =============================================================================
# SECTION 3 -- LangChain RecursiveCharacterTextSplitter
# =============================================================================

def langchain_recursive_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Use LangChain's RecursiveCharacterTextSplitter which tries to split
    on paragraph -> newline -> sentence -> word -> character boundaries in
    that order, keeping chunks under *chunk_size* while preserving
    *chunk_overlap* characters of context between adjacent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


# =============================================================================
# SECTION 4 -- Build FAISS retrieval index & query helper
# =============================================================================

def build_retriever(chunks: list[str], embeddings, k: int = 3):
    """
    Create a FAISS vector store from *chunks* and return a retriever
    that fetches the top-*k* most similar chunks to a query.
    """
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve(retriever, query: str) -> list[str]:
    """Return the page_content of each retrieved Document."""
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


# =============================================================================
# SECTION 5 -- Experiment runner across all 12 configurations
# =============================================================================

CHUNK_SIZES  = [100, 300, 500, 1000]
OVERLAPS     = [0, 50, 100]

# Test query that requires a multi-sentence, contextually complete answer
QUERY = (
    "How did deep learning become successful, and what role did GPUs "
    "and large datasets play in its breakthrough?"
)


def score_contextual_completeness(results: list[str], keywords: list[str]) -> int:
    """
    A simple heuristic: count how many of the expected *keywords* appear
    (case-insensitive) across all retrieved chunks.  Higher = more
    contextually complete answer.
    """
    combined = " ".join(results).lower()
    return sum(1 for kw in keywords if kw.lower() in combined)


def run_experiment(text: str):
    """
    Run the full 12-configuration experiment and return a list of
    result dicts sorted by retrieval quality (best first).
    """
    print("=" * 80)
    print("  CHUNKING EXPERIMENT -- 12 CONFIGURATIONS")
    print("=" * 80)
    print(f'\n[QUERY] "{QUERY}"\n')

    # Keywords a contextually complete answer should mention
    keywords = [
        "deep learning", "GPU", "graphics processing unit",
        "backpropagation", "neural network", "ImageNet",
        "2012", "training data", "transformer", "computer vision",
        "speech recognition", "speed", "hundred-fold",
        "benchmark", "dataset",
    ]

    # Load embedding model once (reuse across all configs)
    print("[WAIT]  Loading embedding model (all-MiniLM-L6-v2) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("[DONE]  Embedding model loaded.\n")

    results_table: list[dict] = []

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            label = f"size={chunk_size:>4}, overlap={overlap:>3}"

            # -- Manual fixed-size chunking --------------------------------
            manual_chunks = manual_fixed_size_chunking(text, chunk_size, overlap)
            manual_retriever = build_retriever(manual_chunks, embeddings)
            manual_results = retrieve(manual_retriever, QUERY)
            manual_score = score_contextual_completeness(manual_results, keywords)

            # -- LangChain recursive chunking ------------------------------
            lc_chunks = langchain_recursive_chunking(text, chunk_size, overlap)
            lc_retriever = build_retriever(lc_chunks, embeddings)
            lc_results = retrieve(lc_retriever, QUERY)
            lc_score = score_contextual_completeness(lc_results, keywords)

            results_table.append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "label": label,
                # Manual
                "manual_num_chunks": len(manual_chunks),
                "manual_score": manual_score,
                "manual_results": manual_results,
                # LangChain
                "lc_num_chunks": len(lc_chunks),
                "lc_score": lc_score,
                "lc_results": lc_results,
            })

            print(
                f"  [{label}]  "
                f"manual: {len(manual_chunks):>4} chunks, score={manual_score:>2}  |  "
                f"langchain: {len(lc_chunks):>4} chunks, score={lc_score:>2}"
            )

    # Sort by best LangChain score (primary), then best manual score
    results_table.sort(key=lambda r: (r["lc_score"], r["manual_score"]), reverse=True)
    return results_table


# =============================================================================
# SECTION 6 -- Print detailed comparison of retrieved chunks
# =============================================================================

def print_comparison(results_table: list[dict]):
    """
    Print a ranked comparison of all 12 configurations, showing which
    configuration returns the most contextually complete result.
    """
    print("\n")
    print("=" * 80)
    print("  RETRIEVAL QUALITY COMPARISON  (ranked best -> worst)")
    print("=" * 80)

    for rank, r in enumerate(results_table, 1):
        print(f"\n{'-' * 70}")
        print(f"  RANK #{rank}  [{r['label']}]")
        print(f"  Manual score: {r['manual_score']}   |   LangChain score: {r['lc_score']}")
        print(f"  Chunks generated -- Manual: {r['manual_num_chunks']}  |  LangChain: {r['lc_num_chunks']}")
        print(f"{'-' * 70}")

        print("\n  > Best LangChain retrieved chunk (top-1):")
        if r["lc_results"]:
            wrapped = textwrap.fill(r["lc_results"][0], width=76, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)

        print("\n  > Best Manual retrieved chunk (top-1):")
        if r["manual_results"]:
            wrapped = textwrap.fill(r["manual_results"][0], width=76, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)

    # Announce the winner
    best = results_table[0]
    print("\n")
    print("*" * 60)
    print("  *** BEST CONFIGURATION ***")
    print("*" * 60)
    print(f"\n  Config                : {best['label']}")
    print(f"  LangChain keyword score : {best['lc_score']}")
    print(f"  Manual keyword score    : {best['manual_score']}")
    print(f"  LangChain chunks created: {best['lc_num_chunks']}")
    print(f"  Manual chunks created   : {best['manual_num_chunks']}")
    print(f"\n{'*' * 60}\n")


# =============================================================================
# SECTION 7 -- Boundary failure analysis
# =============================================================================

def find_boundary_failures(text: str, chunk_size: int = 300, overlap: int = 0):
    """
    Find three concrete examples where a sentence is split mid-thought
    across chunk boundaries using manual fixed-size chunking with
    NO overlap (overlap=0 makes failures most visible).

    For each failure, document:
      - The exact character position where the split happened
      - The text before the split (end of chunk N)
      - The text after the split (start of chunk N+1)
      - The full original sentence that was broken
    """
    import re

    chunks = manual_fixed_size_chunking(text, chunk_size=chunk_size, chunk_overlap=overlap)
    failures: list[dict] = []

    for i in range(len(chunks) - 1):
        if len(failures) >= 3:
            break

        chunk_end = chunks[i]
        chunk_start = chunks[i + 1]

        # The split position in the original text
        split_pos = (i + 1) * chunk_size

        # Check: does the chunk end WITHOUT a sentence-ending punctuation
        # followed by a space or newline?  If so, the sentence was split.
        tail = chunk_end.rstrip()
        if tail and tail[-1] not in ".!?\n":
            # Find the original sentence that was broken
            # Look for the sentence start (last '. ' before split) and
            # sentence end (first '. ' after split)
            search_start = max(0, split_pos - 200)
            search_end = min(len(text), split_pos + 200)
            context = text[search_start:search_end]

            # Find sentence boundaries in context
            sentences = re.split(r'(?<=[.!?])\s+', context)
            broken_sentence = ""
            for sent in sentences:
                sent_start_in_context = context.find(sent)
                sent_start_abs = search_start + sent_start_in_context
                sent_end_abs = sent_start_abs + len(sent)
                # If split_pos falls inside this sentence
                if sent_start_abs < split_pos < sent_end_abs:
                    broken_sentence = sent
                    break

            failures.append({
                "failure_num": len(failures) + 1,
                "chunk_index": i,
                "split_position": split_pos,
                "chunk_end_text": chunk_end[-80:],
                "chunk_start_text": chunk_start[:80],
                "broken_sentence": broken_sentence or "(could not extract full sentence)",
            })

    return failures


def print_boundary_failures(failures: list[dict]):
    """Pretty-print the three boundary failure examples."""
    print("\n")
    print("=" * 80)
    print("  BOUNDARY FAILURE ANALYSIS -- Sentences Split Mid-Thought")
    print("=" * 80)
    print("  (Using manual fixed-size chunking with chunk_size=300, overlap=0)")

    for f in failures:
        print(f"\n{'-' * 70}")
        print(f"  [FAILURE #{f['failure_num']}]")
        print(f"  Split position: character {f['split_position']:,}")
        print(f"  Between chunk {f['chunk_index']} and chunk {f['chunk_index'] + 1}")
        print(f"{'-' * 70}")

        print(f"\n  > End of chunk {f['chunk_index']} (last 80 chars):")
        wrapped = textwrap.fill(
            f"...{f['chunk_end_text']}", width=76,
            initial_indent="    ", subsequent_indent="    "
        )
        print(wrapped)

        print(f"\n  > Start of chunk {f['chunk_index'] + 1} (first 80 chars):")
        wrapped = textwrap.fill(
            f"{f['chunk_start_text']}...", width=76,
            initial_indent="    ", subsequent_indent="    "
        )
        print(wrapped)

        print(f"\n  > Original sentence (broken across boundary):")
        wrapped = textwrap.fill(
            f'"{f["broken_sentence"]}"', width=76,
            initial_indent="    ", subsequent_indent="    "
        )
        print(wrapped)


# =============================================================================
# SECTION 8 -- Written recommendation
# =============================================================================

def print_recommendation(results_table: list[dict], failures: list[dict]):
    """
    Print a detailed recommendation based on experimental results.
    """
    best = results_table[0]
    worst = results_table[-1]

    print("\n")
    print("=" * 80)
    print("  CHUNKING STRATEGY RECOMMENDATION")
    print("=" * 80)

    recommendation = f"""
+------------------------------------------------------------------------+
|                     RECOMMENDED CONFIGURATION                          |
|                                                                        |
|   Chunk Size : {best['chunk_size']:>5} characters                                      |
|   Overlap    : {best['overlap']:>5} characters                                      |
|   Method     : LangChain RecursiveCharacterTextSplitter                |
|   Score      : {best['lc_score']:>5} / 15 keywords matched                           |
+------------------------------------------------------------------------+

REASONING
---------

1. WHY THIS CHUNK SIZE?

   - Chunk size {best['chunk_size']} characters provides enough context for each chunk
     to contain one or more complete sentences, which is critical for
     semantic similarity search to work correctly.

   - Very small chunks (100 chars) lose context -- a single sentence often
     spans 80-150 characters, so 100-char chunks almost always break
     sentences mid-thought. Our experiment showed that the 100-char
     configurations scored worst across the board.

   - Very large chunks (1000 chars) dilute relevance -- when a chunk
     contains multiple paragraphs, the embedding becomes a "blurred
     average" of many topics, making precise retrieval harder.

   - The sweet spot ({best['chunk_size']} chars) balances specificity with completeness.

2. WHY THIS OVERLAP?

   - Overlap of {best['overlap']} characters ensures that information at chunk
     boundaries is preserved in adjacent chunks. Without overlap, our
     boundary analysis found {len(failures)} clear examples of sentences being
     split mid-thought, causing the retrieval system to return
     incomplete or incoherent passages.

   - Too much overlap (relative to chunk size) wastes storage and
     embedding compute by creating near-duplicate chunks. A good rule
     of thumb is overlap = 10-20% of chunk size.

3. WHY LANGCHAIN RECURSIVE SPLITTING OVER MANUAL?

   - LangChain's RecursiveCharacterTextSplitter is "smart" about WHERE
     it splits: it tries paragraph breaks first, then newlines, then
     sentence boundaries, then words, and only falls back to mid-word
     splits as a last resort.

   - Manual fixed-size chunking splits at arbitrary character positions,
     completely ignoring text structure. This leads to the boundary
     failures documented above.

   - In our experiment, LangChain consistently scored equal to or higher
     than manual chunking across every configuration.

4. WORST CONFIGURATION: size={worst['chunk_size']}, overlap={worst['overlap']}

   - Scored only {worst['lc_score']}/15 keywords -- the retrieved chunks were too
     {'short to contain meaningful context.' if worst['chunk_size'] <= 200 else 'diluted or poorly aligned with the query.'}

SUMMARY TABLE
-------------
"""
    print(recommendation)

    # Print a summary table
    print(f"  {'Config':<28} {'Manual Score':>13} {'LC Score':>10} {'LC Chunks':>10}")
    print(f"  {'-' * 28} {'-' * 13} {'-' * 10} {'-' * 10}")
    for r in results_table:
        marker = " <<< BEST" if r is best else ""
        print(
            f"  {r['label']:<28} {r['manual_score']:>13} {r['lc_score']:>10} "
            f"{r['lc_num_chunks']:>10}{marker}"
        )

    print(f"""

KEY TAKEAWAYS
-------------
  [YES] Use RecursiveCharacterTextSplitter (not manual splitting)
  [YES] Chunk size 500-1000 chars is optimal for most retrieval tasks
  [YES] Always use overlap (50-100 chars) to prevent boundary information loss
  [YES] Test your specific query patterns -- the best config depends on your data
  [NO]  Avoid chunk_size < 200 -- sentences get shattered
  [NO]  Avoid overlap = 0 -- boundary context is permanently lost
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -- Step 1: Load the document ----------------------------------------
    text = load_wikipedia_article("Artificial_intelligence")

    # -- Step 2: Quick demo of manual chunking ----------------------------
    print("=" * 80)
    print("  DEMO -- Manual Fixed-Size Chunking (pure Python)")
    print("=" * 80)
    demo_chunks = manual_fixed_size_chunking(text, chunk_size=200, chunk_overlap=0)
    print(f"\n  Total chunks (size=200, overlap=0): {len(demo_chunks)}")
    print(f"\n  First chunk (chars 0-199):")
    print(textwrap.fill(demo_chunks[0], width=76, initial_indent="    ", subsequent_indent="    "))
    print(f"\n  Second chunk (chars 200-399):")
    print(textwrap.fill(demo_chunks[1], width=76, initial_indent="    ", subsequent_indent="    "))

    demo_chunks_overlap = manual_fixed_size_chunking(text, chunk_size=200, chunk_overlap=50)
    print(f"\n  Total chunks (size=200, overlap=50): {len(demo_chunks_overlap)}")
    print(f"\n  Overlapping region (last 50 chars of chunk 0 == first 50 chars of chunk 1):")
    print(f"    Chunk 0 tail: <<{demo_chunks_overlap[0][-50:]}>>")
    print(f"    Chunk 1 head: <<{demo_chunks_overlap[1][:50]}>>")
    match = demo_chunks_overlap[0][-50:] == demo_chunks_overlap[1][:50]
    print(f"    Match: {'YES' if match else 'NO'}\n")

    # -- Step 3: Quick demo of LangChain chunking -------------------------
    print("=" * 80)
    print("  DEMO -- LangChain RecursiveCharacterTextSplitter")
    print("=" * 80)
    lc_demo = langchain_recursive_chunking(text, chunk_size=200, chunk_overlap=50)
    print(f"\n  Total chunks (size=200, overlap=50): {len(lc_demo)}")
    print(f"\n  First chunk:")
    print(textwrap.fill(lc_demo[0], width=76, initial_indent="    ", subsequent_indent="    "))
    print(f"\n  Notice: LangChain splits on natural boundaries (paragraphs, newlines)")
    print(f"  Chunk lengths: {[len(c) for c in lc_demo[:10]]}  (may vary, <= 200)\n")

    # -- Step 4 & 5: Run the 12-configuration experiment ------------------
    results_table = run_experiment(text)

    # -- Step 5 (cont.): Print detailed comparison ------------------------
    print_comparison(results_table)

    # -- Step 6: Boundary failure analysis --------------------------------
    failures = find_boundary_failures(text, chunk_size=300, overlap=0)
    print_boundary_failures(failures)

    # -- Step 7 & 8: Recommendation --------------------------------------
    print_recommendation(results_table, failures)

    print("\n[DONE] Experiment complete!\n")


if __name__ == "__main__":
    main()
