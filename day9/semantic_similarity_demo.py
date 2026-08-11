"""
Day 9 - Semantic Similarity Demo
---------------------------------
In this script, I'm exploring how we can measure how "similar" two
sentences are using TF-IDF vectors and cosine similarity.

The main idea: computers can't understand text the way we do, so we
need to convert words into numbers first. Then we can actually do
math on them to figure out which sentences talk about similar things.

What's covered:
  - Why raw text comparison doesn't work
  - Converting 10 sentences into TF-IDF vectors
  - Building a similarity matrix
  - A handy find_similar() function
  - Testing it with real examples
  - A heatmap to visualise the patterns
  - The math behind cosine similarity (and why it works)
"""

# --- Setting things up ---
import sys
import io

# This fixes a Windows encoding issue with special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Using Agg backend so the script doesn't freeze waiting for a window to close
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -------------------------------------------------------------------------
# Part 1: Why can't we just compare text directly?
# -------------------------------------------------------------------------
# Before jumping into code, let me explain the core problem here.

print("""
=========================================================================
  WHY PLAIN TEXT CAN'T BE COMPARED DIRECTLY
=========================================================================

So here's the thing - computers don't "understand" language. When we
type "The dog barked loudly", the computer just sees a bunch of bytes.
It has no idea what a dog is, what barking sounds like, or why it
matters. It's all just characters to the machine.

This creates a few real problems when we try to compare sentences:

  1. You can't do math on strings
     What does "dog" + "cat" even mean? To Python, it's just
     "dogcat" - concatenation. There's no way to measure how
     "close" two words are just from their characters.

  2. Synonyms are completely invisible
     "Puppy" and "dog" mean almost the same thing to us, but they
     share almost no letters. A simple character comparison would
     say they're totally different words.

  3. Word order changes meaning but looks the same
     "The cat chased the dog" and "The dog chased the cat" are
     almost identical strings but mean very different things.

  4. There's no concept of "distance" between texts
     We need some kind of number line or coordinate system where
     we can place texts and measure gaps between them.

The fix? Turn text into numbers - specifically, into vectors (lists
of numbers). Once we have vectors, we can use all the math tools
we know: dot products, angles, distances, and so on.

That's exactly what TF-IDF does for us.
""")


# -------------------------------------------------------------------------
# Part 2: Our sentence collection (the corpus)
# -------------------------------------------------------------------------
# I picked 10 sentences covering 3 clearly different topics.
# The idea is that sentences about the same topic should end up
# being more "similar" to each other than to sentences from
# different topics.

corpus = [
    # --- Topic A: Animals and Pets ---
    "The dog and the puppy played together in the garden",                  # 0
    "A small puppy dog was adopted from the local animal shelter",          # 1
    "Cats and dogs are the most popular pet animals in the world",          # 2
    "The veterinarian treated the sick dog and a young puppy today",        # 3

    # --- Topic B: Space and Astronomy ---
    "NASA launched a spacecraft to explore a distant planet in space",      # 4
    "The astronaut traveled through space to reach the space station",      # 5
    "Scientists discovered a new planet orbiting a distant star in space",  # 6

    # --- Topic C: Cooking and Food ---
    "The chef used fresh ingredients to cook a delicious pasta dish",       # 7
    "She followed the recipe to cook a spicy chicken dish with herbs",      # 8
    "Cooking a healthy dish with vegetables requires the right recipe",     # 9
]

# Shorter labels for the heatmap (full sentences won't fit on the axes)
labels = [
    "A0: dog puppy garden",
    "A1: puppy dog shelter",
    "A2: cats dogs pets",
    "A3: vet dog puppy",
    "B0: NASA planet space",
    "B1: astronaut space",
    "B2: planet star space",
    "C0: chef cook pasta",
    "C1: recipe cook dish",
    "C2: cooking dish recipe",
]

print("=" * 70)
print("  OUR CORPUS - 10 sentences across 3 topics")
print("=" * 70)
for i, sent in enumerate(corpus):
    print(f"  [{i}] {sent}")
print()


# -------------------------------------------------------------------------
# Part 3: Converting text to TF-IDF vectors
# -------------------------------------------------------------------------
# TF-IDF stands for Term Frequency - Inverse Document Frequency.
# Basically, it gives higher weight to words that are important in
# a specific document but rare across all documents. Common words
# like "the" and "a" get filtered out (that's what stop_words does).

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",   # removes common words that don't carry meaning
    norm="l2",              # normalises each vector to unit length
)

tfidf_matrix = vectorizer.fit_transform(corpus)

print("=" * 70)
print("  TF-IDF VECTORISATION RESULTS")
print("=" * 70)
print(f"  Number of documents : {tfidf_matrix.shape[0]}")
print(f"  Vocabulary size     : {tfidf_matrix.shape[1]} unique terms")
print(f"  Non-zero entries    : {tfidf_matrix.nnz}")

terms = vectorizer.get_feature_names_out()
print(f"\n  First 15 terms in the vocabulary:")
print(f"  {list(terms[:15])}")
print()

# So now each of our 10 sentences is represented as a vector with
# one dimension per unique word. Most values are 0 (sparse) because
# each sentence only uses a few words from the whole vocabulary.


# -------------------------------------------------------------------------
# Part 4: Computing the similarity matrix
# -------------------------------------------------------------------------
# Now for the fun part - let's see how similar each pair of sentences is.
# cosine_similarity gives us a 10x10 matrix where entry [i][j] tells
# us how similar sentence i is to sentence j.

sim_matrix = cosine_similarity(tfidf_matrix)

print("=" * 70)
print("  COSINE SIMILARITY MATRIX (10 x 10)")
print("=" * 70)
print("  Each value shows how similar two sentences are (0 = nothing")
print("  in common, 1 = identical topic words)")
print()

header = "      " + "  ".join(f"[{i:>2}]" for i in range(10))
print(header)
for i in range(10):
    row_str = "  ".join(f"{sim_matrix[i, j]:.3f}" for j in range(10))
    print(f"  [{i:>2}] {row_str}")
print()

# If you look at this carefully, you'll notice that sentences within
# the same topic (like 0-3 for animals) tend to have higher scores
# with each other than with sentences from other topics. That's
# exactly what we'd expect!


# -------------------------------------------------------------------------
# Part 5: The find_similar() function
# -------------------------------------------------------------------------
# This is probably the most useful part. Given any new sentence, this
# function finds the closest matches in our corpus.

def find_similar(query, corpus, top_k=3, vectorizer=None, tfidf_matrix=None):
    """
    Takes a new sentence and finds the most similar ones from the corpus.

    How it works:
    1. Convert the query into a TF-IDF vector using the same vocabulary
    2. Calculate cosine similarity with every corpus sentence
    3. Return the top-k highest scoring matches

    Returns a list of (score, index, sentence) tuples.
    """
    # If we don't have pre-fitted objects, create them fresh
    if vectorizer is None or tfidf_matrix is None:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", norm="l2")
        tfidf_matrix = vectorizer.fit_transform(corpus)

    # Transform the query using the SAME vocabulary as the corpus
    # (this is important - we need matching dimensions)
    query_vec = vectorizer.transform([query])

    # Get similarity scores against all corpus sentences
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Sort by score (highest first) and pick the top results
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((float(scores[idx]), int(idx), corpus[idx]))

    return results


# -------------------------------------------------------------------------
# Part 6: Testing it out!
# -------------------------------------------------------------------------
# Let's throw some queries at our function and see what comes back.
# I'm testing with sentences that should match specific topics,
# plus a couple of completely unrelated ones to see what happens.

test_queries = [
    # These should match our corpus topics pretty well
    "The little dog ran across the yard",
    "A new puppy was adopted from the shelter",
    "Astronomers discovered a new exoplanet",
    "She cooked a delicious meal with herbs and spices",

    # These are deliberately off-topic to test edge cases
    "The airplane flew over the mountain range",
    "Stock markets crashed due to economic uncertainty",
]

print("=" * 70)
print("  TESTING find_similar() - Top 3 matches for each query")
print("=" * 70)

for q in test_queries:
    print(f"\n  Query: \"{q}\"")
    results = find_similar(q, corpus, top_k=3,
                           vectorizer=vectorizer,
                           tfidf_matrix=tfidf_matrix)
    for rank, (score, idx, sent) in enumerate(results, 1):
        print(f"    Match #{rank}  (score: {score:.4f})  [{idx}] {sent}")

print()

# Pretty cool - the dog/puppy queries correctly find the animal
# sentences, the space query finds the astronomy sentences, and the
# cooking query finds the food sentences. The off-topic queries
# (airplane, stock market) get zeros across the board because they
# share no meaningful words with our corpus.


# -------------------------------------------------------------------------
# Part 7: Head-to-head comparisons (dog vs puppy, dog vs airplane)
# -------------------------------------------------------------------------
# This is where it gets interesting. Let's directly compare specific
# pairs to see the difference between semantically similar and
# dissimilar sentences.

print("=" * 70)
print("  HEAD-TO-HEAD COMPARISONS")
print("=" * 70)

pairs = [
    ("The dog barked loudly",     "The puppy played outside"),
    ("The dog barked loudly",     "The airplane flew at high altitude"),
    ("A playful puppy ran fast",  "A cute dog ran quickly"),
    ("NASA launched a rocket",    "She baked a chocolate cake"),
]

pair_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", norm="l2")

for s1, s2 in pairs:
    pair_matrix = pair_vectorizer.fit_transform([s1, s2])
    score = cosine_similarity(pair_matrix[0:1], pair_matrix[1:2])[0, 0]

    if score > 0.1:
        verdict = "SIMILAR"
    else:
        verdict = "NOT SIMILAR"

    print(f"  [{verdict}]  score = {score:.4f}")
    print(f"      Sentence A: \"{s1}\"")
    print(f"      Sentence B: \"{s2}\"")
    print()

# One thing to notice: "dog" vs "puppy" shows as NOT SIMILAR here
# because TF-IDF treats them as completely different words. It only
# looks at exact word matches, not meaning. This is a known limitation -
# more advanced models like Word2Vec or BERT would catch that relationship.


# -------------------------------------------------------------------------
# Part 8: Visualising the similarity matrix as a heatmap
# -------------------------------------------------------------------------
# Numbers in a table are hard to read. A heatmap makes the patterns
# jump out immediately - you can literally SEE the topic clusters.

fig, ax = plt.subplots(figsize=(10, 8))

# Warmer colours = higher similarity
cax = ax.imshow(sim_matrix, cmap="YlOrRd", interpolation="nearest",
                vmin=0, vmax=1)

# Add a colour scale on the side
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Cosine Similarity", fontsize=12)

# Label the axes with our short sentence descriptions
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

# Write the actual score inside each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        val = sim_matrix[i, j]
        text_color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold")

# Draw dashed blue boxes around each topic cluster so the grouping is obvious
for start, end in [(0, 4), (4, 7), (7, 10)]:
    rect = mpatches.Rectangle(
        (start - 0.5, start - 0.5), end - start, end - start,
        linewidth=2.5, edgecolor="blue", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)

ax.set_title("Cosine Similarity Heatmap\n(TF-IDF Vectors across 3 Topic Clusters)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Sentence", fontsize=11)
ax.set_ylabel("Sentence", fontsize=11)

plt.tight_layout()
plt.savefig("day9_similarity_heatmap.png", dpi=150, bbox_inches="tight")
print("  Heatmap saved as day9_similarity_heatmap.png")
print()

# The heatmap really makes it obvious - there are three bright clusters
# along the diagonal, one for each topic. The off-diagonal areas are
# pale/cold, meaning sentences from different topics have little in common.


# -------------------------------------------------------------------------
# Part 9: Why cosine similarity uses ANGLE, not distance
# -------------------------------------------------------------------------
# This is the conceptual part that ties everything together.

print("""
=========================================================================
  THE MATH BEHIND COSINE SIMILARITY
=========================================================================

  So why do we use cosine similarity instead of just measuring the
  straight-line (Euclidean) distance between vectors? It comes down
  to one key insight: we care about DIRECTION, not MAGNITUDE.

  The formula:

                        A . B               sum(ai * bi)
    cos(theta) = --------------- = ----------------------------
                  ||A|| x ||B||    sqrt(sum(ai^2)) * sqrt(bi^2)

  What this calculates is the cosine of the angle between two vectors.

    - cos(theta) = 1.0  -->  vectors point the same way (very similar)
    - cos(theta) = 0.0  -->  vectors are perpendicular  (unrelated)
    - cos(theta) = -1.0 -->  vectors point opposite ways (but this
                              doesn't happen with TF-IDF since all
                              values are non-negative)

  Why does angle matter more than distance for text?
  --------------------------------------------------

  1. LENGTH INDEPENDENCE
     Imagine two articles about dogs - one is 100 words, the other
     is 10,000 words. Their TF-IDF vectors point in roughly the same
     direction (both are "about dogs") but have very different lengths.

     Euclidean distance would say: "these are far apart!" (wrong)
     Cosine similarity would say: "these point the same way!" (right)

  2. WORKS WELL IN HIGH DIMENSIONS
     Our vocabulary has dozens of dimensions, and real-world text can
     have thousands. In high-dimensional spaces, Euclidean distances
     become less meaningful (everything seems equally far apart).
     Cosine similarity doesn't suffer from this problem.

  3. EASY TO INTERPRET
     The result is always between 0 and 1 (for TF-IDF vectors).
     0.0 means "completely unrelated", 1.0 means "identical topic".
     Euclidean distance could be anything from 0 to infinity, which
     makes it harder to set thresholds.

  4. SIMPLE INTUITION
     Think of it this way:
       - Cosine asks: "Are these arrows pointing the same direction?"
       - Euclidean asks: "How far apart are the tips of these arrows?"

     For text comparison, direction = topic, magnitude = document length.
     Since we care about topic (not length), angle is what matters.
""")


# -------------------------------------------------------------------------
# Part 10: A quick numerical example to prove the point
# -------------------------------------------------------------------------
# Let's make this concrete with actual numbers.

print("=" * 70)
print("  PROVING IT WITH NUMBERS")
print("=" * 70)

A = np.array([1, 2])
B = np.array([100, 200])
C = np.array([2, 1])

def euclidean_dist(x, y):
    return np.linalg.norm(x - y)

def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(f"\n  Let's say we have three tiny 'documents':")
print(f"    A = {list(A)}       (a short article about dogs)")
print(f"    B = {list(B)}   (a LONG article about dogs - same word ratio)")
print(f"    C = {list(C)}       (a short article about cats - different ratio)")

print(f"\n  Comparing A and B (same topic, different length):")
print(f"    Euclidean distance = {euclidean_dist(A, B):.2f}  <-- huge! but they're about the same thing")
print(f"    Cosine similarity  = {cosine_sim(A, B):.4f}    <-- perfect match, as expected")

print(f"\n  Comparing A and C (different topic, same length):")
print(f"    Euclidean distance = {euclidean_dist(A, C):.2f}   <-- tiny! but they're about different things")
print(f"    Cosine similarity  = {cosine_sim(A, C):.4f}    <-- different direction, lower score")

print(f"""
  What this tells us:
    Euclidean distance got it backwards - it said A and B are far
    apart (they're not, same topic) and A and C are close (they're
    not, different topics).

    Cosine similarity got it right both times. That's why we use it
    for text comparison.
""")

print("=" * 70)
print("  Done! Day 9 - Semantic Similarity Demo complete.")
print("=" * 70)
