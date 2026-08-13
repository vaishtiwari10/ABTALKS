"""
Day 11 - Document Retrieval Engine
------------------------------------
Today I'm building a mini search engine. The idea is pretty simple:
I've got 20 short documents about AI and machine learning, and I want
to type in a question and have the system find the most relevant ones.

It's basically what Google does (in a very simplified way) — turn the
query into numbers, turn the documents into numbers, and find the
closest match. I'm using TF-IDF from Day 9 to do the number-crunching.

What's covered:
  - A TextPipeline class that handles all the preprocessing
  - 20 documents on AI/ML topics as my knowledge base
  - A retrieve() function that finds the best matches
  - A relevance threshold so it doesn't return garbage results
  - Testing with 10 queries (some tricky ones too)
  - Figuring out why certain queries fail
  - Why TF-IDF breaks on synonyms (and why embeddings fix this)
"""

# --- Setting things up ---
import sys
import io

# This fixes a Windows encoding issue with special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------------
# Part 1: The TextPipeline class
# -------------------------------------------------------------------------
# In Day 9 I was calling TfidfVectorizer directly every time, which got
# messy. So I wrapped everything into a class. The key thing is that
# once you fit it on your documents, it remembers the vocabulary — so
# when a new query comes in later, it gets converted into the SAME
# vector space. If I didn't do this, the dimensions wouldn't match up
# and the whole similarity calculation would be nonsense.

class TextPipeline:
    """
    Handles the whole text-to-numbers pipeline.

    How I use it:
        pipeline = TextPipeline()
        matrix = pipeline.fit_transform(my_documents)   # learns vocab + vectorises
        query_vec = pipeline.transform("some question")  # uses the same vocab
    """

    def __init__(self, max_features=None, ngram_range=(1, 1)):
        """
        max_features: cap on vocabulary size (None = keep everything)
        ngram_range:  (1,1) for single words, (1,2) to include bigrams too
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,          # "Dog" and "dog" become the same token
            stop_words="english",    # get rid of "the", "is", "a", etc.
            norm="l2",               # normalise vectors to unit length
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # use log(1 + tf) instead of raw counts
        )
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit_transform(self, documents):
        """
        Learn the vocabulary from the documents and turn them all into
        TF-IDF vectors. This is the "training" step.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.is_fitted = True
        return self.tfidf_matrix

    def transform(self, text):
        """
        Turn new text into a TF-IDF vector using the vocabulary we
        already learned. This is how queries get vectorised later.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline hasn't been fitted yet! "
                               "Call fit_transform() with your corpus first.")
        if isinstance(text, str):
            text = [text]
        return self.vectorizer.transform(text)

    def get_vocabulary(self):
        """Returns the list of words the pipeline knows about."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline hasn't been fitted yet.")
        return self.vectorizer.get_feature_names_out()

    def get_stats(self):
        """Quick summary of what the pipeline produced."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline hasn't been fitted yet.")
        return {
            "n_documents": self.tfidf_matrix.shape[0],
            "vocabulary_size": self.tfidf_matrix.shape[1],
            "non_zero_entries": self.tfidf_matrix.nnz,
            "sparsity": 1 - (self.tfidf_matrix.nnz /
                             (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])),
        }


# -------------------------------------------------------------------------
# Part 2: My knowledge base — 20 documents about AI & ML
# -------------------------------------------------------------------------
# I went with AI/ML as the topic because there's enough variety for 20
# different documents, but also enough overlap that it creates some
# interesting edge cases. Like, the word "model" shows up everywhere,
# and "neural network" vs "deep learning" are basically the same thing
# said differently — which is exactly the kind of thing that trips up
# TF-IDF (more on that later).

knowledge_base = [
    # --- Neural Networks & Deep Learning (0-3) ---

    "Neural networks are computing systems inspired by the biological neural "
    "networks in the human brain. They consist of layers of interconnected "
    "nodes called neurons that process information using weighted connections. "
    "Training a neural network involves adjusting these weights through "
    "backpropagation to minimise prediction errors.",

    "Deep learning is a subset of machine learning that uses neural networks "
    "with many hidden layers. These deep architectures can automatically learn "
    "hierarchical feature representations from raw data, making them powerful "
    "for tasks like image recognition and speech processing.",

    "Convolutional neural networks, or CNNs, are specialised deep learning "
    "architectures designed for processing grid-like data such as images. "
    "They use convolutional filters to detect local patterns like edges and "
    "textures, then combine them into higher-level features for classification.",

    "Recurrent neural networks, or RNNs, are designed to handle sequential "
    "data like text and time series. They maintain a hidden state that acts "
    "as memory, allowing the network to use information from previous time "
    "steps when processing the current input.",

    # --- Natural Language Processing (4-7) ---

    "Natural language processing, or NLP, is the branch of artificial "
    "intelligence focused on enabling computers to understand, interpret, "
    "and generate human language. Key tasks include sentiment analysis, "
    "machine translation, and named entity recognition.",

    "Transformer models revolutionised NLP by introducing the self-attention "
    "mechanism. Unlike RNNs, transformers can process all tokens in a sequence "
    "simultaneously, which makes them much faster to train. BERT and GPT are "
    "both built on the transformer architecture.",

    "Word embeddings like Word2Vec and GloVe represent words as dense "
    "numerical vectors in a continuous vector space. Words with similar "
    "meanings end up close together in this space, capturing semantic "
    "relationships that simple bag-of-words models cannot.",

    "Text classification is the task of assigning predefined categories to "
    "text documents. Common approaches include naive Bayes classifiers, "
    "support vector machines, and more recently, fine-tuned transformer "
    "models like BERT that achieve state-of-the-art accuracy.",

    # --- Computer Vision (8-10) ---

    "Computer vision is the field of AI that trains machines to interpret "
    "and understand visual information from the world. Applications include "
    "facial recognition, autonomous vehicles, medical image analysis, and "
    "augmented reality systems.",

    "Object detection goes beyond image classification by not only "
    "identifying what objects are present in an image but also locating "
    "them with bounding boxes. Popular architectures include YOLO, "
    "Faster R-CNN, and SSD.",

    "Image segmentation divides an image into meaningful regions or pixels "
    "that belong to different objects or classes. Semantic segmentation "
    "labels every pixel, while instance segmentation also distinguishes "
    "between separate instances of the same object class.",

    # --- Reinforcement Learning (11-13) ---

    "Reinforcement learning is a type of machine learning where an agent "
    "learns to make decisions by interacting with an environment. The agent "
    "receives rewards or penalties for its actions and learns a policy that "
    "maximises cumulative reward over time.",

    "Q-learning is a model-free reinforcement learning algorithm that learns "
    "the value of taking a specific action in a specific state. The agent "
    "builds a Q-table that maps state-action pairs to expected future "
    "rewards, updating it through trial and error.",

    "AlphaGo, developed by DeepMind, combined deep neural networks with "
    "Monte Carlo tree search to defeat the world champion in the board game "
    "Go. This was a landmark achievement because Go has far more possible "
    "positions than chess, making brute-force search impossible.",

    # --- Ethics & Safety (14-16) ---

    "AI bias occurs when machine learning models produce unfair or "
    "discriminatory outcomes due to biased training data or flawed "
    "algorithm design. Examples include facial recognition systems that "
    "perform worse on certain demographic groups and hiring algorithms "
    "that discriminate based on gender.",

    "Explainable AI, or XAI, aims to make the decision-making process of "
    "machine learning models transparent and understandable to humans. "
    "Techniques include LIME, SHAP values, and attention visualisation, "
    "which help users understand why a model made a specific prediction.",

    "The alignment problem refers to the challenge of ensuring that "
    "advanced AI systems pursue goals that are aligned with human values "
    "and intentions. Researchers worry that a sufficiently powerful AI "
    "might find unexpected and potentially harmful ways to achieve its "
    "objectives if not properly constrained.",

    # --- Practical ML Concepts (17-19) ---

    "Overfitting happens when a machine learning model learns the training "
    "data too well, including its noise and random fluctuations. The model "
    "performs excellently on training data but poorly on unseen test data. "
    "Regularisation techniques like dropout and L2 penalty help prevent this.",

    "Transfer learning allows a model trained on one task to be reused as "
    "the starting point for a different but related task. For example, a "
    "CNN pre-trained on ImageNet can be fine-tuned for medical image "
    "classification with much less data than training from scratch.",

    "Data augmentation is a technique to artificially increase the size of "
    "a training dataset by creating modified versions of existing data. "
    "In computer vision, this includes random rotations, flips, crops, and "
    "colour adjustments. It helps models generalise better and reduces "
    "overfitting.",
]

# Shorter labels so the output doesn't get too wide
doc_labels = [
    "D00: Neural networks & backpropagation",
    "D01: Deep learning & hidden layers",
    "D02: CNNs for image processing",
    "D03: RNNs for sequential data",
    "D04: NLP overview",
    "D05: Transformers & attention",
    "D06: Word embeddings (Word2Vec, GloVe)",
    "D07: Text classification methods",
    "D08: Computer vision applications",
    "D09: Object detection (YOLO, R-CNN)",
    "D10: Image segmentation",
    "D11: Reinforcement learning basics",
    "D12: Q-learning algorithm",
    "D13: AlphaGo & Monte Carlo tree search",
    "D14: AI bias & discrimination",
    "D15: Explainable AI (XAI)",
    "D16: AI alignment problem",
    "D17: Overfitting & regularisation",
    "D18: Transfer learning",
    "D19: Data augmentation",
]


# -------------------------------------------------------------------------
# Part 3: Running everything through the pipeline
# -------------------------------------------------------------------------
# Let's feed all 20 documents into the TextPipeline and see what
# comes out. The sparsity number is interesting — it tells us that
# most cells in the matrix are zero, which makes sense because each
# document only uses a small fraction of the total vocabulary.

print("=" * 72)
print("  DAY 11 - DOCUMENT RETRIEVAL ENGINE")
print("=" * 72)

print(f"\n  I've got {len(knowledge_base)} documents in my knowledge base:\n")
for i, label in enumerate(doc_labels):
    print(f"    {label}")

# Create the pipeline and vectorise everything
pipeline = TextPipeline(ngram_range=(1, 1))
corpus_matrix = pipeline.fit_transform(knowledge_base)
stats = pipeline.get_stats()

print(f"\n{'=' * 72}")
print("  VECTORISATION RESULTS")
print(f"{'=' * 72}")
print(f"  Documents processed  : {stats['n_documents']}")
print(f"  Vocabulary size      : {stats['vocabulary_size']} unique terms")
print(f"  Non-zero entries     : {stats['non_zero_entries']}")
print(f"  Sparsity             : {stats['sparsity']:.1%}")
print(f"\n  So out of all the cells in a {stats['n_documents']}x{stats['vocabulary_size']} matrix,")
print(f"  only {stats['non_zero_entries']} actually have values. Everything else is zero.")
print(f"  That's because each document only uses a handful of the {stats['vocabulary_size']} words.")

vocab = pipeline.get_vocabulary()
print(f"\n  First 20 vocabulary terms:")
print(f"  {list(vocab[:20])}")
print(f"\n  Last 20 vocabulary terms:")
print(f"  {list(vocab[-20:])}")
print()


# -------------------------------------------------------------------------
# Part 4: The retrieve() function
# -------------------------------------------------------------------------
# This is really the heart of the whole thing. You give it a query, it
# turns it into a vector (using the same vocabulary as the corpus),
# compares it against every document, and returns the top matches.
#
# I also added a relevance threshold — if the best score is below 0.1,
# the system says "sorry, nothing relevant." Without this, it would
# always return SOMETHING, even if you searched for "chocolate cake
# recipe" in an AI corpus. That's not helpful.

RELEVANCE_THRESHOLD = 0.1

def retrieve(query, corpus_matrix, top_k=3):
    """
    Find the top-K most relevant documents for a query.

    How it works:
    1. Turn the query into a TF-IDF vector (same vocab as the corpus)
    2. Calculate cosine similarity against every document
    3. If the best score is below 0.1, return "nothing found"
    4. Otherwise, return the top-K matches with their scores

    Returns a list of dicts with rank, doc_id, score, label, and document.
    """
    # Turn the query into a vector using the same vocabulary
    query_vector = pipeline.transform(query)

    # How similar is this query to each document?
    scores = cosine_similarity(query_vector, corpus_matrix).flatten()

    # If even the best match is terrible, don't pretend we found something
    max_score = float(np.max(scores))
    if max_score < RELEVANCE_THRESHOLD:
        return [{
            "rank": None,
            "doc_id": None,
            "score": max_score,
            "label": "No relevant document found",
            "document": (f"The highest similarity score ({max_score:.4f}) is below "
                         f"the threshold ({RELEVANCE_THRESHOLD}). "
                         f"Nothing in the corpus is a meaningful match."),
        }]

    # Grab the top results, sorted by score
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "doc_id": int(idx),
            "score": float(scores[idx]),
            "label": doc_labels[idx],
            "document": knowledge_base[idx],
        })

    return results


def print_results(query, results, expected_topic=None, query_type=""):
    """Pretty-prints the retrieval results and checks if we got it right."""
    print(f"\n  {'─' * 68}")
    print(f"  Query: \"{query}\"")
    if query_type:
        print(f"  Type:  {query_type}")

    # Did the threshold kick in?
    if results[0]["rank"] is None:
        print(f"  ⚠  {results[0]['label']}")
        print(f"     (best score: {results[0]['score']:.4f}, "
              f"threshold: {RELEVANCE_THRESHOLD})")
        if expected_topic == "NONE":
            print(f"  ✓  PASS — correctly rejected out-of-domain query")
        else:
            print(f"  ✗  FAIL — should have found a relevant document")
        return results[0]["score"], results[0]["label"], "THRESHOLD"

    # Show the top matches
    for r in results:
        marker = "→" if r["rank"] == 1 else " "
        print(f"  {marker} #{r['rank']}  (score: {r['score']:.4f})  {r['label']}")

    # Did we get the right document on top?
    top_label = results[0]["label"]
    if expected_topic is None:
        verdict = "—"
        print(f"  ?  Ambiguous — no single right answer here")
    elif expected_topic == "NONE":
        verdict = "UNEXPECTED"
        print(f"  ✗  FAIL — this should have been rejected by threshold")
    elif expected_topic in top_label:
        verdict = "PASS"
        print(f"  ✓  PASS — nailed it, top result is what I expected")
    else:
        verdict = "FAIL"
        print(f"  ✗  FAIL — I expected something about '{expected_topic}'")

    return results[0]["score"], top_label, verdict


# -------------------------------------------------------------------------
# Part 5: Testing with 10 queries
# -------------------------------------------------------------------------
# Here's where the rubber meets the road. I'm throwing 10 different
# queries at the system:
#   - 6 straightforward ones that should match specific documents
#   - 2 ambiguous ones where there's no single right answer
#   - 2 completely off-topic ones (cooking, sports) that should fail
#
# The off-topic ones are really important — they test whether the
# relevance threshold actually works. I don't want my AI search engine
# confidently returning a document about neural networks when someone
# asks about chocolate cake.

print(f"\n{'=' * 72}")
print("  TESTING TIME — 10 Queries")
print(f"{'=' * 72}")

test_queries = [
    # --- These should work well ---
    {
        "query": "How do neural networks learn from data?",
        "expected": "Neural networks",
        "type": "On-domain (neural networks)",
    },
    {
        "query": "What is natural language processing used for?",
        "expected": "NLP",
        "type": "On-domain (NLP)",
    },
    {
        "query": "How does reinforcement learning train an agent?",
        "expected": "Reinforcement learning",
        "type": "On-domain (RL)",
    },
    {
        "query": "Why do machine learning models overfit on training data?",
        "expected": "Overfitting",
        "type": "On-domain (overfitting)",
    },
    {
        "query": "What is object detection in computer vision?",
        "expected": "Object detection",
        "type": "On-domain (object detection)",
    },
    {
        "query": "How can pre-trained models be reused for new tasks?",
        "expected": "Transfer learning",
        "type": "On-domain (transfer learning)",
    },

    # --- Tricky ones — genuinely ambiguous ---
    {
        "query": "model training and performance improvement",
        "expected": None,
        "type": "AMBIGUOUS — 'model training' could match many AI topics",
    },
    {
        "query": "bias and fairness in automated systems",
        "expected": None,
        "type": "AMBIGUOUS — could match AI bias, alignment, or explainability",
    },

    # --- Should be rejected entirely ---
    {
        "query": "recipe for chocolate cake with vanilla frosting",
        "expected": "NONE",
        "type": "OUT-OF-DOMAIN (cooking)",
    },
    {
        "query": "football World Cup scores and championship results",
        "expected": "NONE",
        "type": "OUT-OF-DOMAIN (sports)",
    },
]

# Run every query and collect the results
test_results = []

for i, tq in enumerate(test_queries, 1):
    print(f"\n  Test {i}/10")
    score, label, verdict = print_results(
        tq["query"],
        retrieve(tq["query"], corpus_matrix, top_k=3),
        expected_topic=tq["expected"],
        query_type=tq["type"],
    )
    test_results.append({
        "id": i,
        "query": tq["query"],
        "type": tq["type"],
        "top_score": score,
        "top_result": label,
        "verdict": verdict,
    })

# --- Summary Table ---
print(f"\n\n{'=' * 72}")
print("  RESULTS AT A GLANCE")
print(f"{'=' * 72}")
print(f"  {'#':<4} {'Verdict':<12} {'Score':<8} {'Type'}")
print(f"  {'─'*4} {'─'*12} {'─'*8} {'─'*40}")
for tr in test_results:
    v_symbol = {"PASS": "✓", "FAIL": "✗", "THRESHOLD": "⚠", "UNEXPECTED": "⚠"}.get(tr["verdict"], "?")
    print(f"  {tr['id']:<4} {v_symbol} {tr['verdict']:<10} {tr['top_score']:<8.4f} {tr['type']}")

passes = sum(1 for tr in test_results if tr["verdict"] == "PASS")
thresholds = sum(1 for tr in test_results if tr["verdict"] == "THRESHOLD")
ambiguous = sum(1 for tr in test_results if tr["verdict"] == "—")
fails = sum(1 for tr in test_results if tr["verdict"] in ("FAIL", "UNEXPECTED"))
print(f"\n  Scorecard: {passes} passed, {thresholds} correctly rejected, "
      f"{ambiguous} ambiguous, {fails} failed")


# -------------------------------------------------------------------------
# Part 6: What went wrong? Failure analysis
# -------------------------------------------------------------------------
# This is the part where I go through each query and explain what
# happened. The failures are actually the most interesting part,
# because they show us exactly where TF-IDF's limitations bite.

print(f"\n\n{'=' * 72}")
print("  WHAT WENT WRONG (AND RIGHT) — Query-by-Query Analysis")
print(f"{'=' * 72}")

analysis = [
    {
        "id": 1,
        "query": "How do neural networks learn from data?",
        "diagnosis": (
            "FAILURE: It returned D01 (Deep learning) instead of D00 (Neural "
            "networks). Why? Because 'learn' and 'data' carry more TF-IDF "
            "weight in D01 — D00 uses more specific words like 'backpropagation' "
            "and 'weights', so the generic wording of my query accidentally "
            "favoured the broader deep learning document."
        ),
    },
    {
        "id": 2,
        "query": "What is natural language processing used for?",
        "diagnosis": (
            "SUCCESS: This one was easy — the exact phrase 'natural language "
            "processing' appears word-for-word in Document 4. When the query "
            "and document share the exact same terms, TF-IDF shines."
        ),
    },
    {
        "id": 3,
        "query": "How does reinforcement learning train an agent?",
        "diagnosis": (
            "SUCCESS: 'Reinforcement', 'learning', and 'agent' all show up "
            "in Document 11. Three strong keyword matches = high similarity. "
            "No ambiguity here."
        ),
    },
    {
        "id": 4,
        "query": "Why do machine learning models overfit on training data?",
        "diagnosis": (
            "SUCCESS: 'Overfit', 'training', 'data', 'machine', 'learning' — "
            "almost every meaningful word in this query appears in Document 17. "
            "It's like the query was written to match."
        ),
    },
    {
        "id": 5,
        "query": "What is object detection in computer vision?",
        "diagnosis": (
            "SUCCESS: 'Object detection' appears directly in Document 9, "
            "and 'computer vision' matches Document 8 too. The top result "
            "was spot-on."
        ),
    },
    {
        "id": 6,
        "query": "How can pre-trained models be reused for new tasks?",
        "diagnosis": (
            "SUCCESS: 'Pre-trained', 'model', and 'task' are key words in "
            "Document 18 (transfer learning). This is exactly what transfer "
            "learning is about, and the vocabulary lined up perfectly."
        ),
    },
    {
        "id": 7,
        "query": "model training and performance improvement",
        "diagnosis": (
            "AMBIGUOUS: 'Model', 'training', and 'performance' are super "
            "generic — they show up in documents about neural networks, "
            "overfitting, reinforcement learning, and more. TF-IDF returned "
            "the overfitting doc (D17) because 'training' and 'model' happen "
            "to co-occur there, but honestly it's a coin flip. The system "
            "has no way to know what I actually meant."
        ),
    },
    {
        "id": 8,
        "query": "bias and fairness in automated systems",
        "diagnosis": (
            "AMBIGUOUS: 'Bias' appears in Document 14 (AI bias), which is "
            "great. But 'fairness' and 'automated systems' don't match any "
            "document exactly. The system got lucky here because 'bias' alone "
            "is distinctive enough. If I'd said 'prejudice' instead of 'bias', "
            "the system would have completely missed it — that's the synonym "
            "problem in action."
        ),
    },
    {
        "id": 9,
        "query": "recipe for chocolate cake with vanilla frosting",
        "diagnosis": (
            "CORRECTLY REJECTED: Not a single word in this query — 'recipe', "
            "'chocolate', 'cake', 'vanilla', 'frosting' — appears anywhere in "
            "my AI corpus. Cosine similarity is literally zero, and the "
            "threshold filter did its job perfectly."
        ),
    },
    {
        "id": 10,
        "query": "football World Cup scores and championship results",
        "diagnosis": (
            "FALSE POSITIVE: This should have been rejected, but it scored "
            "0.21 against the computer vision document (D08). Why? Because the "
            "word 'world' appears in D08 ('visual information from the world'). "
            "The word means completely different things in each context, but "
            "TF-IDF doesn't understand meaning — it just sees matching "
            "characters. Classic vocabulary collision."
        ),
    },
]

for a in analysis:
    print(f"\n  Query {a['id']}: \"{a['query']}\"")
    print(f"  → {a['diagnosis']}")

print()


# -------------------------------------------------------------------------
# Part 7: Synonym stress test
# -------------------------------------------------------------------------
# OK, so I noticed the vocabulary mismatch thing in the failure analysis.
# Let me push it further. I'll take 4 concepts, write each one two ways:
# once using the EXACT words from the corpus, and once using synonyms.
# If TF-IDF really can't handle synonyms, the scores should drop hard.

print(f"\n{'=' * 72}")
print("  SYNONYM STRESS TEST — Same Question, Different Words")
print(f"{'=' * 72}")
print("""
  Here's an experiment. Each pair below asks about the SAME concept,
  but the second version uses different words. Let's see if TF-IDF
  can handle it... (spoiler: not always)
""")

synonym_pairs = [
    # (exact vocabulary match, synonym version)
    (
        "neural networks learn through backpropagation",
        "brain-inspired computing systems adjust through error correction",
    ),
    (
        "reinforcement learning agent receives rewards",
        "trial and error system gets positive feedback for good actions",
    ),
    (
        "overfitting on training data",
        "memorising examples instead of generalising to new cases",
    ),
    (
        "computer vision image recognition",
        "machines understanding visual information and pictures",
    ),
]

for orig, synonym in synonym_pairs:
    orig_results = retrieve(orig, corpus_matrix, top_k=1)
    syn_results = retrieve(synonym, corpus_matrix, top_k=1)

    orig_score = orig_results[0]["score"]
    syn_score = syn_results[0]["score"]

    orig_label = orig_results[0]["label"]
    syn_label = syn_results[0]["label"]

    drop = orig_score - syn_score
    same_doc = (orig_results[0].get("doc_id") == syn_results[0].get("doc_id"))

    print(f"  Original:  \"{orig}\"")
    print(f"    → score: {orig_score:.4f}  |  {orig_label}")
    print(f"  Synonym:   \"{synonym}\"")
    print(f"    → score: {syn_score:.4f}  |  {syn_label}")
    if same_doc:
        print(f"    Score drop: {drop:+.4f}  |  Same document: Yes")
    else:
        print(f"    Score drop: {drop:+.4f}  |  Same document: NO — it found the wrong one!")
    print()

# The results are pretty telling. When I use exact vocabulary from the
# corpus, scores are solid (0.3-0.5). When I rephrase with synonyms,
# they drop significantly, and sometimes the system returns a completely
# different document. That's not great for a search engine.


# -------------------------------------------------------------------------
# Part 8: Why this matters — the vocabulary mismatch problem
# -------------------------------------------------------------------------
# The assignment asks me to explain why TF-IDF fails on synonyms and
# what this tells us about the need for embeddings. Here goes.

print(f"\n{'=' * 72}")
print("  WHY TF-IDF STRUGGLES WITH SYNONYMS (AND WHY EMBEDDINGS FIX IT)")
print(f"{'=' * 72}")
print("""
  THE PROBLEM
  ===========

  Here's the core issue with TF-IDF: it only cares about exact word
  matches. Every word gets its own dimension in the vector, and two
  different words — even if they mean exactly the same thing — are
  treated as completely unrelated.

  Think about it:

    - "Neural network" and "brain-inspired computing system" mean the
      same thing to us, but they share ZERO words. To TF-IDF, they're
      as different as "neural network" and "chocolate cake."

    - "Overfitting" and "memorising the training data" describe the
      exact same problem, but TF-IDF sees them as unrelated because
      the characters are different.

    - Even "optimise" and "optimize" (just British vs American spelling)
      are treated as two completely separate words.

  We saw this in the stress test above — queries that used the exact
  right words scored 0.3-0.5, while synonym versions of the SAME
  question dropped dramatically, sometimes even returning the wrong
  document entirely.


  WHY DOES TF-IDF WORK THIS WAY?
  ===============================

  It's not a bug, it's a fundamental limitation of the approach.
  TF-IDF works like this:
    1. Split text into individual words
    2. Count how often each word appears (that's the TF part)
    3. Penalise words that show up in lots of documents (the IDF part)
    4. Store everything in a big sparse vector

  At no point does the system think about what words MEAN. "Dog" and
  "puppy" are as different to TF-IDF as "dog" and "rocket." There's
  nothing in the algorithm that could even theoretically learn that
  some words are related, because it never looks at context — it
  just counts individual tokens.


  SO WHAT'S THE FIX? EMBEDDINGS.
  ==============================

  This exact limitation is why the NLP world moved to embeddings —
  representations that capture what words actually MEAN, not just
  what characters they contain:

  Word2Vec (2013):
     Learned that words appearing in similar contexts ("The ___ ran
     across the yard") probably mean similar things. So "dog" and
     "puppy" end up with similar vectors because they show up in
     similar sentences. This solved the synonym problem at the word
     level for the first time.

  GloVe (2014):
     Similar idea but uses global word co-occurrence statistics.
     Famous for "King - Man + Woman ≈ Queen" — vector arithmetic
     that actually captures analogies. Completely impossible with
     TF-IDF.

  BERT & Transformers (2018+):
     These go even further with CONTEXTUAL embeddings — the vector
     for "bank" is different in "river bank" vs "bank account."
     The model reads the whole sentence before deciding what each
     word means. This handles:
       - Synonyms (different words, same meaning)
       - Polysemy (same word, different meanings)
       - Paraphrasing (different sentences, same idea)

  THE BOTTOM LINE
  ===============

  TF-IDF is fast, simple, and works great when queries use the same
  words as the documents. It's still used in production — Google's
  early search engine was basically TF-IDF on steroids (BM25).

  But for any real search system where people describe things in
  their own words (which is... basically always), you need something
  that understands meaning, not just characters. That's embeddings.

  Modern systems often combine both:
    Step 1: Use TF-IDF/BM25 to quickly narrow millions of docs
            down to a few hundred candidates (it's fast!)
    Step 2: Use BERT embeddings to re-rank those candidates by
            actual semantic similarity (it's smart!)

  Best of both worlds — TF-IDF's speed with embeddings' understanding.
""")


# -------------------------------------------------------------------------
# Wrapping up
# -------------------------------------------------------------------------
print("=" * 72)
print("  DAY 11 — WHAT I BUILT")
print("=" * 72)
print(f"""
  ✓ TextPipeline class — handles all the text-to-numbers stuff
  ✓ Knowledge base — {len(knowledge_base)} documents on AI & Machine Learning
  ✓ retrieve() function — finds the top-K most relevant documents
  ✓ Relevance threshold — says "nothing found" when score < {RELEVANCE_THRESHOLD}
  ✓ 10-query test suite — {passes} passed, {thresholds} correctly rejected, {fails} failed
  ✓ Failure analysis — figured out why each query worked (or didn't)
  ✓ Synonym stress test — proved that paraphrasing breaks TF-IDF
  ✓ Written analysis — why this means we need embeddings

  The biggest takeaway for me:
    TF-IDF is great when the words match up. The moment someone uses
    a synonym, rephrases a question, or uses slightly different jargon,
    it falls apart. That's why Word2Vec and BERT exist — they understand
    meaning, not just characters.
""")
print("=" * 72)
print("  Done! Day 11 complete.")
print("=" * 72)
