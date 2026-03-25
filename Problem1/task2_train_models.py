"""
Task 2: Word2Vec FROM SCRATCH — CBOW & Skip-gram with Negative Sampling
CSL 7640 - Assignment 2, Problem 1

"""

import numpy as np
import json
import os
import time
import collections

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)



# 1.  VOCABULARY
class Vocabulary:
    """
    Maps tokens ↔ integer indices.
    Builds a negative-sampling table: freq^(3/4) distribution,
    stored as a flat int array for O(1) sampling.
    """

    def __init__(self, min_count=2):
        self.min_count = min_count
        self.word2idx  = {}
        self.idx2word  = []
        self.word_freq = {}
        self.neg_table = None

    def build(self, sentences):
        counter = collections.Counter(
            tok for sent in sentences for tok in sent
        )
        kept = {w: c for w, c in counter.items() if c >= self.min_count}
        for word, freq in sorted(kept.items(), key=lambda x: -x[1]):
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.word_freq[word] = freq

        self._build_neg_table()
        print(f"[Vocab] {len(self.idx2word)} tokens kept "
              f"(min_count={self.min_count}, "
              f"{len(counter) - len(kept)} pruned)")

    def _build_neg_table(self, table_size=1_000_000):
        """
        Pre-compute 1M-slot lookup table.
        Slots proportional to freq^(3/4) — down-weights stop words.
        """
        freqs   = np.array([self.word_freq[w] for w in self.idx2word], dtype=np.float64)
        powered = freqs ** 0.75
        probs   = powered / powered.sum()
        counts  = np.maximum((probs * table_size).astype(int), 1)
        self.neg_table = np.repeat(np.arange(len(self.idx2word), dtype=np.int32), counts)
        np.random.shuffle(self.neg_table)

    def sample_negatives_batch(self, exclude_idxs, k):
        """
        Sample k negatives for each of B positive pairs at once.
        Returns array of shape (B, k).

        Strategy: draw B*k*2 candidates, reshape to (B, k*2),
        then for each row keep first k entries != its exclude index.
        Fallback: if any row has < k valid negatives after filtering,
        fill remaining slots with random draws (rare edge case).
        """
        B = len(exclude_idxs)
        # Draw a large pool at once — single randint call
        pool = self.neg_table[
            np.random.randint(0, len(self.neg_table), size=B * k * 3)
        ].reshape(B, k * 3)

        result = np.zeros((B, k), dtype=np.int32)
        for i in range(B):
            # Filter out the positive word for this row
            valid = pool[i][pool[i] != exclude_idxs[i]]
            if len(valid) >= k:
                result[i] = valid[:k]
            else:
                # Rare: just fill with random indices (not the positive)
                while len(valid) < k:
                    extra = self.neg_table[np.random.randint(0, len(self.neg_table))]
                    if extra != exclude_idxs[i]:
                        valid = np.append(valid, extra)
                result[i] = valid[:k]

        return result  # (B, k)

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word):
        return word in self.word2idx



# 2.  WORD2VEC MODEL
class Word2Vec:
    """
    Word2Vec — CBOW and Skip-gram with Negative Sampling.
    dependency: NumPy.

    Parameters
    embed_dim  : word vector dimensionality
    window     : context window radius (each side)
    negative   : number of negative samples per positive pair
    sg         : 0 = CBOW, 1 = Skip-gram
    lr         : initial learning rate (linearly decayed to lr*0.0001)
    epochs     : training passes over corpus
    min_count  : minimum word frequency
    seed       : random seed
    """

    def __init__(self, embed_dim=100, window=5, negative=5,
                 sg=0, lr=0.025, epochs=5, min_count=2, seed=42):
        self.embed_dim = embed_dim
        self.window    = window
        self.negative  = negative
        self.sg        = sg
        self.lr        = lr
        self.epochs    = epochs
        self.min_count = min_count
        self.seed      = seed

        self.vocab      = None
        self.W_in       = None   # (V, D) input  embeddings
        self.W_out      = None   # (V, D) output embeddings
        self.total_loss = []

    # Init 
    def _init_weights(self, V):
        rng = np.random.default_rng(self.seed)
        self.W_in  = (rng.random((V, self.embed_dim)) - 0.5) / self.embed_dim
        self.W_out = np.zeros((V, self.embed_dim))

    # Sigmoid (array-safe) 
    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    
    # SKIP-GRAM: build all pairs for one sentence, then do ONE batch update
    

    def _collect_skipgram_pairs(self, indices):
        """
        Return two int arrays: center_arr, context_arr
        Each entry is one (center, context) training pair.
        Window is randomly shrunk per center word (original paper trick).
        """
        centers  = []
        contexts = []
        n = len(indices)
        for i, c in enumerate(indices):
            win = np.random.randint(1, self.window + 1)
            for j in range(max(0, i - win), min(n, i + win + 1)):
                if j != i:
                    centers.append(c)
                    contexts.append(indices[j])

        return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)

    def _train_skipgram_batch(self, indices, lr):
        """
        VECTORIZED skip-gram update for one sentence.

        Steps:
          1. Collect all (center, context) pairs → arrays of shape (B,)
          2. Sample B*K negatives in one shot → shape (B, K)
          3. Gather embeddings:
               H     = W_in[centers]       (B, D)
               V_pos = W_out[contexts]     (B, D)
               V_neg = W_out[neg_idxs]    (B, K, D)
          4. Compute all scores at once:
               pos_score = rowwise_dot(H, V_pos)          (B,)
               neg_score = einsum('bkd,bd->bk', V_neg, H) (B, K)
          5. Compute sigmoid errors:
               e_pos = σ(pos_score) - 1   (B,)   ← label=1
               e_neg = σ(neg_score)       (B, K) ← label=0
          6. Compute gradients w.r.t. H:
               grad_H  = e_pos[:,None]*V_pos + (e_neg[:,:,None]*V_neg).sum(axis=1)
          7. Update W_out using np.add.at (handles repeated indices):
               W_out[contexts] -= lr * e_pos[:,None] * H
               W_out[neg_idxs] -= lr * e_neg[:,:,None] * H[:,None,:]
          8. Update W_in:
               W_in[centers]   -= lr * grad_H
        """
        if len(indices) < 2:
            return 0.0

        centers, contexts = self._collect_skipgram_pairs(indices)
        if len(centers) == 0:
            return 0.0

        B = len(centers)
        K = self.negative

        # Step 2: sample all negatives at once
        neg_idxs = self.vocab.sample_negatives_batch(contexts, K)  # (B, K)

        # Step 3: gather embeddings
        H     = self.W_in[centers]         # (B, D)
        V_pos = self.W_out[contexts]       # (B, D)
        V_neg = self.W_out[neg_idxs]      # (B, K, D)

        # Step 4: scores
        pos_score = (H * V_pos).sum(axis=1)                          # (B,)
        neg_score = np.einsum('bkd,bd->bk', V_neg, H)               # (B, K)

        # Step 5: sigmoid errors
        e_pos = self._sigmoid(pos_score) - 1.0                       # (B,)
        e_neg = self._sigmoid(neg_score)                             # (B, K)

        # Step 6: loss (for logging)
        loss = (-np.log(self._sigmoid(pos_score) + 1e-10).sum()
                - np.log(1.0 - self._sigmoid(neg_score) + 1e-10).sum())

        # Step 7: gradient w.r.t. H
        # from positive:  e_pos[:, None] * V_pos     → (B, D)
        # from negatives: (e_neg[:,:,None]*V_neg).sum(1) → (B, D)
        grad_H = e_pos[:, None] * V_pos + (e_neg[:, :, None] * V_neg).sum(axis=1)

        # Step 8: update W_out (positive)
        # np.add.at handles duplicate center/context indices correctly
        np.add.at(self.W_out, contexts, -lr * e_pos[:, None] * H)

        # Update W_out (negatives) — neg_idxs is (B, K), flatten for add.at
        neg_grads = lr * e_neg[:, :, None] * H[:, None, :]  # (B, K, D)
        np.add.at(self.W_out, neg_idxs.ravel(),
                  -neg_grads.reshape(-1, self.embed_dim))

        # Update W_in
        np.add.at(self.W_in, centers, -lr * grad_H)

        return float(loss)


    # CBOW: build all (center, context_group) items, then batch update
    

    def _train_cbow_batch(self, indices, lr):
        """
        VECTORIZED CBOW update for one sentence.

        For CBOW the input representation h_i = mean(W_in[context_i]).
        Each center word i has a different context set, so we can't use a
        single matrix gather. We process each center separately BUT vectorize
        the NS loss across ALL (center, negative) pairs using einsum.

        Steps per center i:
          h_i    = W_in[ctx_i].mean(axis=0)          (D,)
          V_pos  = W_out[center_i]                   (D,)
          V_neg  = W_out[neg_i]                      (K, D)
          scores computed vectorially over K negatives in one shot
          grad_h distributed back to all context words equally
        """
        if len(indices) < 2:
            return 0.0

        n    = len(indices)
        K    = self.negative
        loss = 0.0

        for i, center_idx in enumerate(indices):
            win = np.random.randint(1, self.window + 1)
            ctx_idxs = np.array([
                indices[j]
                for j in range(max(0, i - win), min(n, i + win + 1))
                if j != i
            ], dtype=np.int32)
            if len(ctx_idxs) == 0:
                continue

            # h = mean of context vectors
            h = self.W_in[ctx_idxs].mean(axis=0)  # (D,)

            # Sample K negatives
            neg_idxs = self.vocab.sample_negatives_batch(
                np.array([center_idx], dtype=np.int32), K
            )[0]   # (K,)

            # Positive score
            V_pos     = self.W_out[center_idx]              # (D,)
            pos_score = np.dot(V_pos, h)                    # scalar
            sigma_pos = self._sigmoid(pos_score)
            e_pos     = sigma_pos - 1.0

            # Negative scores — vectorized over K
            V_neg     = self.W_out[neg_idxs]               # (K, D)
            neg_score = V_neg @ h                           # (K,)
            sigma_neg = self._sigmoid(neg_score)            # (K,)
            e_neg     = sigma_neg                           # (K,)

            loss += (-np.log(sigma_pos + 1e-10)
                     - np.log(1.0 - sigma_neg + 1e-10).sum())

            # Gradient w.r.t. h
            grad_h = e_pos * V_pos + (e_neg[:, None] * V_neg).sum(axis=0)  # (D,)

            # Update W_out
            self.W_out[center_idx] -= lr * e_pos * h
            np.add.at(self.W_out, neg_idxs, -lr * e_neg[:, None] * h[None, :])

            # Distribute grad equally to all context words in W_in
            np.add.at(self.W_in, ctx_idxs,
                      -lr * (grad_h / len(ctx_idxs))[None, :])

        return loss

    #FIT
    def fit(self, sentences):
        """
        Build vocab, init weights, train with linear LR decay.
        Sentences are pre-converted to int32 numpy arrays for speed.
        """
        self.vocab = Vocabulary(min_count=self.min_count)
        self.vocab.build(sentences)
        V = len(self.vocab)
        self._init_weights(V)

        # Pre-convert to numpy int arrays — avoids dict lookups during training
        indexed = []
        for sent in sentences:
            arr = np.array(
                [self.vocab.word2idx[w] for w in sent if w in self.vocab.word2idx],
                dtype=np.int32
            )
            if len(arr) >= 2:
                indexed.append(arr)

        model_type  = "Skip-gram" if self.sg else "CBOW"
        total_steps = self.epochs * len(indexed)
        print(f"\n[{model_type}] vocab={V} | embed={self.embed_dim} | "
              f"win={self.window} | neg={self.negative} | epochs={self.epochs}")
        print(f"  {len(indexed)} usable sentences")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            np.random.shuffle(indexed)
            t0 = time.time()

            for step, idx_sent in enumerate(indexed):
                # Linear LR decay: lr → lr*0.0001
                global_step = epoch * len(indexed) + step
                progress    = global_step / max(total_steps - 1, 1)
                current_lr  = max(self.lr * 0.0001, self.lr * (1.0 - progress))

                if self.sg == 1:
                    epoch_loss += self._train_skipgram_batch(idx_sent, current_lr)
                else:
                    epoch_loss += self._train_cbow_batch(idx_sent, current_lr)

            avg = epoch_loss / max(len(indexed), 1)
            self.total_loss.append(avg)
            print(f"  Epoch {epoch+1}/{self.epochs} | "
                  f"loss={avg:.4f} | lr={current_lr:.6f} | {time.time()-t0:.1f}s")

        print(f"[{model_type}] Training complete.")

      #INFERENCE

    def get_vector(self, word):
        if word not in self.vocab:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.W_in[self.vocab.word2idx[word]].copy()

    def similarity(self, word1, word2):
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        return float(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def most_similar(self, word, topn=5):
        """Vectorized cosine similarity against all vocab vectors."""
        if word not in self.vocab:
            raise KeyError(f"'{word}' not in vocabulary")
        qv     = self.get_vector(word)
        qv    /= np.linalg.norm(qv) + 1e-10
        norms  = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        sims   = (self.W_in / norms) @ qv
        sims[self.vocab.word2idx[word]] = -2.0
        top    = np.argpartition(sims, -topn)[-topn:]
        top    = top[np.argsort(sims[top])[::-1]]
        return [(self.vocab.idx2word[i], round(float(sims[i]), 4)) for i in top]

    def analogy(self, positive, negative, topn=5):
        """3CosAdd: vec(D) ≈ vec(B) - vec(A) + vec(C)"""
        query     = np.zeros(self.embed_dim)
        all_words = set(positive + negative)
        for w in positive:
            v = self.get_vector(w); query += v / (np.linalg.norm(v) + 1e-10)
        for w in negative:
            v = self.get_vector(w); query -= v / (np.linalg.norm(v) + 1e-10)
        query /= np.linalg.norm(query) + 1e-10
        norms  = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        sims   = (self.W_in / norms) @ query
        for w in all_words:
            if w in self.vocab:
                sims[self.vocab.word2idx[w]] = -2.0
        top = np.argpartition(sims, -topn)[-topn:]
        top = top[np.argsort(sims[top])[::-1]]
        return [(self.vocab.idx2word[i], round(float(sims[i]), 4)) for i in top]

    #Save and load
    def save(self, path):
        np.savez(path + ".npz", W_in=self.W_in, W_out=self.W_out)
        meta = {
            "embed_dim":  self.embed_dim,
            "window":     self.window,
            "negative":   self.negative,
            "sg":         self.sg,
            "lr":         self.lr,
            "epochs":     self.epochs,
            "min_count":  self.min_count,
            "word2idx":   self.vocab.word2idx,
            "idx2word":   self.vocab.idx2word,
            "word_freq":  self.vocab.word_freq,
            "total_loss": self.total_loss,
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f)
        print(f"[Saved] {path}.npz + {path}.json")

    @classmethod
    def load(cls, path):
        with open(path + ".json") as f:
            meta = json.load(f)
        arrays = np.load(path + ".npz")
        model  = cls(
            embed_dim=meta["embed_dim"], window=meta["window"],
            negative=meta["negative"],  sg=meta["sg"],
            lr=meta["lr"],              epochs=meta["epochs"],
            min_count=meta["min_count"],
        )
        model.vocab           = Vocabulary()
        model.vocab.word2idx  = meta["word2idx"]
        model.vocab.idx2word  = meta["idx2word"]
        model.vocab.word_freq = meta["word_freq"]
        model.vocab._build_neg_table()
        model.W_in        = arrays["W_in"]
        model.W_out       = arrays["W_out"]
        model.total_loss  = meta["total_loss"]
        return model

#Hyperparameter grid+runner
SIMILAR_PAIRS = [
    ("research", "project"),
    ("student",  "phd"),
    ("faculty",  "professor"),
    ("exam",     "grade"),
    ("department", "course"),
]

def intrinsic_score(model):
    scores = []
    for w1, w2 in SIMILAR_PAIRS:
        try:
            scores.append(model.similarity(w1, w2))
        except KeyError:
            pass
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def load_corpus(path):
    sentences = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sentences.append(toks)
    print(f"[Corpus] {len(sentences)} sentences loaded.")
    return sentences


def train_all():
    # Auto-detect corpus — works with corpus.txt OR data/cleaned_corpus.txt
    candidates = [
        "corpus.txt",                                  # manually collected file
        os.path.join(DATA_DIR, "cleaned_corpus.txt"), # preprocessed output
        "cleaned_corpus.txt",
    ]
    corpus_path = next((c for c in candidates if os.path.exists(c)), None)

    if corpus_path is None:
        print("[ERROR] No corpus file found. Tried:")
        for c in candidates:
            print(f"         {c}")
        return

    print(f"[Corpus] Using: {corpus_path}")
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    sentences = load_corpus(corpus_path)
    results   = {"cbow": [], "skipgram": []}

    CBOW_CONFIGS = [
        dict(embed_dim=50,  window=3, negative=5,  sg=0, epochs=5),
        dict(embed_dim=100, window=5, negative=5,  sg=0, epochs=5),
        dict(embed_dim=200, window=7, negative=5,  sg=0, epochs=5),
    ]
    SG_CONFIGS = [
        dict(embed_dim=50,  window=3, negative=5,  sg=1, epochs=5),
        dict(embed_dim=100, window=5, negative=5,  sg=1, epochs=5),
        dict(embed_dim=100, window=5, negative=10, sg=1, epochs=5),
        dict(embed_dim=200, window=7, negative=10, sg=1, epochs=5),
    ]

    best_cbow_score, best_cbow = -1, None
    best_sg_score,   best_sg   = -1, None

    print("\n" + "="*60)
    print("TRAINING CBOW (vectorized, NumPy only)")
    print("="*60)
    for i, cfg in enumerate(CBOW_CONFIGS):
        model = Word2Vec(**cfg, lr=0.025, min_count=2, seed=42)
        model.fit(sentences)
        score = intrinsic_score(model)
        print(f"  >> CBOW-{i+1} score: {score}\n")
        results["cbow"].append({**cfg, "score": score})
        if score > best_cbow_score:
            best_cbow_score, best_cbow = score, model

    best_cbow.save(os.path.join(MODELS_DIR, "cbow_best"))

    print("\n" + "="*60)
    print("TRAINING SKIP-GRAM (vectorized, NumPy only)")
    print("="*60)
    for i, cfg in enumerate(SG_CONFIGS):
        model = Word2Vec(**cfg, lr=0.025, min_count=2, seed=42)
        model.fit(sentences)
        score = intrinsic_score(model)
        print(f"  >> SG-{i+1} score: {score}\n")
        results["skipgram"].append({**cfg, "score": score})
        if score > best_sg_score:
            best_sg_score, best_sg = score, model

    best_sg.save(os.path.join(MODELS_DIR, "skipgram_best"))

    out = "training_results.json"  # save next to corpus.txt for simplicity
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")

    print("\n" + "="*55)
    print(f"{'Model':<12} {'Dim':>5} {'Win':>5} {'Neg':>5} {'Score':>8}")
    print("-"*40)
    for tag, rows in [("CBOW", results["cbow"]), ("Skip-gram", results["skipgram"])]:
        for r in rows:
            print(f"{tag:<12} {r['embed_dim']:>5} {r['window']:>5} "
                  f"{r['negative']:>5} {r['score']:>8.4f}")


if __name__ == "__main__":
    train_all()