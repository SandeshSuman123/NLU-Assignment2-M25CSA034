"""
Task 3: Semantic Analysis — Nearest Neighbors & Analogy Experiments
CSL 7640 - Assignment 2, Problem 1

"""

import os
import sys
import json

#  Import Word2Vec created 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task2_train_models import Word2Vec

# path
MODELS_DIR = "models"


#configuration


QUERY_WORDS = ["research", "student", "phd", "exam", "course"]

# Analogy experiments
# Format: (A, B, C, expected_answer, description)
# The model must find D such that  A : B :: C : D
# Method: D = most_similar(positive=[B, C], negative=[A])
ANALOGIES = [
    # Required by assignment
    ("ug",      "btech",         "pg",          "mtech/msc",
     "UG : BTech :: PG : ?"),

    # Academic role analogy
    ("student", "study",         "faculty",     "teach/research",
     "student : study :: faculty : ?"),

    # Degree level analogy
    ("btech",   "undergraduate", "mtech",       "postgraduate",
     "BTech : undergraduate :: MTech : ?"),

    # Academic output analogy
    ("student", "assignment",    "researcher",  "paper/publication",
     "student : assignment :: researcher : ?"),

    # Evaluation analogy
    ("student", "exam",          "faculty",     "evaluation/assessment",
     "student : exam :: faculty : ?"),
]


#load models
def load_models():
    """Load best CBOW and Skip-gram models """
    cbow_path = os.path.join(MODELS_DIR, "cbow_best")
    sg_path   = os.path.join(MODELS_DIR, "skipgram_best")

    if not os.path.exists(cbow_path + ".json"):
        return None, None

    cbow = Word2Vec.load(cbow_path)
    sg   = Word2Vec.load(sg_path)

    print(f"[Loaded] CBOW      — vocab={len(cbow.vocab)}, "
          f"dim={cbow.embed_dim}, window={cbow.window}")
    print(f"[Loaded] Skip-gram — vocab={len(sg.vocab)}, "
          f"dim={sg.embed_dim}, window={sg.window}, neg={sg.negative}")
    return cbow, sg

#part 1- nearest neighbors
def get_nearest_neighbors(model, words, topn=5):
    """
    For each query word, compute cosine similarity against all vocabulary
    vectors and return the top-n most similar words.

    cosine_sim(u, v) = dot(u, v) / (||u|| * ||v||)

    Returns a dict: {word -> [(neighbor, similarity), ...] or "OOV" string}
    """
    results = {}
    for word in words:
        if word in model.vocab:
            # model.most_similar() does vectorized cosine over entire W_in matrix
            neighbors = model.most_similar(word, topn=topn)
            results[word] = neighbors
        else:
            results[word] = None   # out of vocabulary
    return results


def print_nn_results(nn_cbow, nn_sg, query_words):
    """Print nearest neighbor results as a clean side-by-side table."""

    print("\n" + "═"*78)
    print("  NEAREST NEIGHBORS — Top-5 by Cosine Similarity")
    print("═"*78)
    print(f"  {'Query':<12} {'Rank':<5} {'CBOW Neighbor (sim)':^30} {'Skip-gram Neighbor (sim)':^30}")
    print("─"*78)

    for word in query_words:
        c_res = nn_cbow.get(word)
        s_res = nn_sg.get(word)

        # Handle OOV
        if c_res is None and s_res is None:
            print(f"  {word:<12}  [OOV in both models — word not in corpus]")
            print()
            continue

        for i in range(5):
            # Format CBOW neighbor
            if c_res is None:
                c_str = "[OOV]"
            elif i < len(c_res):
                c_str = f"{c_res[i][0]}  ({c_res[i][1]:+.4f})"
            else:
                c_str = "-"

            # Format Skip-gram neighbor
            if s_res is None:
                s_str = "[OOV]"
            elif i < len(s_res):
                s_str = f"{s_res[i][0]}  ({s_res[i][1]:+.4f})"
            else:
                s_str = "-"

            word_label = word if i == 0 else ""
            rank_label = f"#{i+1}"
            print(f"  {word_label:<12} {rank_label:<5} {c_str:<30} {s_str:<30}")

        print()  # blank line between words

#part 2- Analogy experiments
def run_analogies(model, analogies, model_label):
    """
    Run 3CosAdd analogy experiments.

    For each (A, B, C):
      query = normalize(vec(B)) - normalize(vec(A)) + normalize(vec(C))
      answer = word with highest cosine_sim(word_vec, query)
               excluding A, B, C themselves

    Returns list of result dicts for saving to JSON.
    """
    results = []

    print(f"\n  [{model_label}]")
    print(f"  {'Analogy':<40} {'Top Answer':<18} {'Score':>7}  {'Expected'}")
    print(f"  {'─'*40} {'─'*18} {'─'*7}  {'─'*15}")

    for (A, B, C, expected, description) in analogies:
        entry = {
            "analogy":     description,
            "A": A, "B": B, "C": C,
            "expected":    expected,
            "model":       model_label,
        }

        # Check for out-of-vocabulary words
        oov = [w for w in [A, B, C] if w not in model.vocab]
        if oov:
            entry["status"]  = "OOV"
            entry["top5"]    = []
            entry["answer"]  = "N/A"
            print(f"  {description:<40} {'SKIPPED — OOV: ' + str(oov)}")
        else:
            # 3CosAdd: positive=[B, C], negative=[A]
            top5 = model.analogy(positive=[B, C], negative=[A], topn=5)
            entry["status"]  = "OK"
            entry["answer"]  = top5[0][0]
            entry["score"]   = top5[0][1]
            entry["top5"]    = top5

            ans_str = f"{top5[0][0]}"
            print(f"  {description:<40} {ans_str:<18} {top5[0][1]:>7.4f}  {expected}")

        results.append(entry)

    return results


def print_analogy_top5(results):
    """Print full top-5 breakdown for each analogy."""
    print("\n  Full Top-5 Breakdown:")
    print("  " + "─"*60)
    for r in results:
        if r["status"] == "OOV":
            continue
        print(f"\n  {r['analogy']}")
        for rank, (word, score) in enumerate(r["top5"], 1):
            marker = " ◄ best" if rank == 1 else ""
            print(f"    #{rank}: {word:<20} (cosine sim = {score:+.4f}){marker}")


#MAIN
def analyze():
    print("=" * 78)
    print("  TASK 3: SEMANTIC ANALYSIS")
    print("  CSL 7640 — Assignment 2, Problem 1")
    print("=" * 78)

    #  Load models 
    cbow, sg = load_models()
    if cbow is None:
        return

    # Part 1: Nearest Neighbors
    print("\n" + "─"*78)
    print("  PART 1: TOP-5 NEAREST NEIGHBORS")
    print("─"*78)
    print("  Method: cosine_sim(u,v) = dot(u,v) / (||u|| × ||v||)")
    print("  Higher score = more semantically similar\n")

    nn_cbow = get_nearest_neighbors(cbow, QUERY_WORDS, topn=5)
    nn_sg   = get_nearest_neighbors(sg,   QUERY_WORDS, topn=5)
    print_nn_results(nn_cbow, nn_sg, QUERY_WORDS)

    # Part 2: Analogies 
    print("\n" + "─"*78)
    print("  PART 2: ANALOGY EXPERIMENTS  (A : B :: C : ?)")
    print("─"*78)
    print("  Method: 3CosAdd — query = norm(vec(B)) - norm(vec(A)) + norm(vec(C))")
    print("  Find word D (≠ A,B,C) with highest cosine similarity to query\n")

    analogy_cbow = run_analogies(cbow, ANALOGIES, "CBOW")
    print()
    analogy_sg   = run_analogies(sg,   ANALOGIES, "Skip-gram")

    # Full top-5 for each model
    print("\n" + "─"*78)
    print("  CBOW — Full Top-5 per Analogy")
    print_analogy_top5(analogy_cbow)

    print("\n" + "─"*78)
    print("  Skip-gram — Full Top-5 per Analogy")
    print_analogy_top5(analogy_sg)

    # Part 3: Discussion 
    print_discussion(nn_cbow, nn_sg, analogy_cbow, analogy_sg)

    #  Save results to JSON 
    # Convert None (OOV) to string for JSON serialization
    def serialize_nn(nn_dict):
        return {
            w: (res if res is not None else "OOV — not in vocabulary")
            for w, res in nn_dict.items()
        }

    output = {
        "nearest_neighbors": {
            "cbow":     serialize_nn(nn_cbow),
            "skipgram": serialize_nn(nn_sg),
        },
        "analogies": {
            "cbow":     analogy_cbow,
            "skipgram": analogy_sg,
        },
    }

    out_path = "semantic_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] Results → {out_path}")
    print("\n[Done] Task 3 complete.")


if __name__ == "__main__":
    analyze()
