# Task 4: PCA and t-SNE visualization for CBOW and Skip-gram

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from task2_train_models import Word2Vec   # import trained model

# create folder to save images
os.makedirs("visualizations", exist_ok=True)

# define word groups for visualization
clusters = [
    ("Academic", "red", ["ug","pg","phd","btech","mtech","degree"]),
    ("People", "blue", ["student","faculty","professor","researcher"]),
    ("Research", "green", ["research","project","thesis","paper"]),
    ("Assessment", "orange", ["exam","grade","marks","assignment","score"]),
    ("Infra", "purple", ["lab","library","campus","department","institute"])
]

# get vectors for selected words
def get_vectors(model):
    words, vecs, colors = [], [], []
    for name, color, word_list in clusters:
        for w in word_list:
            if w in model.vocab:
                words.append(w)
                vecs.append(model.get_vector(w))
                colors.append(color)
    return words, np.array(vecs), colors

# plot function
def plot(words, coords, colors, title, filename):
    plt.figure(figsize=(8,6))
    plt.scatter(coords[:,0], coords[:,1], c=colors)

    # annotate each word
    for i, w in enumerate(words):
        plt.text(coords[i,0], coords[i,1], w, fontsize=8)

    plt.title(title)
    plt.grid(True)
    plt.savefig("visualizations/" + filename)
    plt.close()

# PCA visualization
def run_pca(model, name):
    words, vecs, colors = get_vectors(model)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)
    plot(words, coords, colors, f"PCA - {name}", f"pca_{name}.png")

# t-SNE visualization
def run_tsne(model, name):
    words, vecs, colors = get_vectors(model)
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    coords = tsne.fit_transform(vecs)
    plot(words, coords, colors, f"t-SNE - {name}", f"tsne_{name}.png")

# main function
def main():
    # load trained models
    cbow = Word2Vec.load("models/cbow_best")
    skipgram = Word2Vec.load("models/skipgram_best")

    # generate plots
    run_pca(cbow, "cbow")
    run_pca(skipgram, "skipgram")

    run_tsne(cbow, "cbow")
    run_tsne(skipgram, "skipgram")

    print("All plots saved in 'visualizations' folder")

if __name__ == "__main__":
    main()