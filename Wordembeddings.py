import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the high-res map
print("Loading semantic space...")
nlp = spacy.load("en_core_web_lg")

def get_analogy_result(a, b, c):
    vec_a, vec_b, vec_c = nlp(a).vector, nlp(b).vector, nlp(c).vector
    target_v = (vec_b - vec_a) + vec_c
    
    # Simple nearest neighbor lookup
    target_v = target_v.reshape(1, -1)
    hashes, _, _ = nlp.vocab.vectors.most_similar(target_v, n=1)
    return nlp.vocab.strings[hashes[0][0]]

def plot_vibe_map(word_groups):
    """
    word_groups: a list of tuples like [("man", "king"), ("woman", "queen")]
    """
    all_words = [word for group in word_groups for word in group]
    vectors = [nlp(w).vector for w in all_words]
    
    # PCA: The Linear Algebra trick to squash 300D -> 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 7))
    
    # Plot the points and draw arrows for the relationships
    for i in range(0, len(all_words), 2):
        p1, p2 = coords[i], coords[i+1]
        plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='blue')
        plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], 
                  head_width=0.05, length_includes_head=True, alpha=0.3)
        plt.annotate(all_words[i], (p1[0], p1[1]), xytext=(5, 5), textcoords='offset points')
        plt.annotate(all_words[i+1], (p2[0], p2[1]), xytext=(5, 5), textcoords='offset points')

    plt.title("The Linear Logic of AI (Parallelism in Vector Space)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- RUN IT ---
# We use pairs to see if the 'jumps' are parallel
pairs = [
    ("uncle", "aunt"),
    ("brother", "sister"),
    ("man", "king"),
    ("woman", "queen")
]

print("Visualizing the 'Gender' and 'Royalty' vectors...")
plot_vibe_map(pairs)