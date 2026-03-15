# Word Embeddings and Vector Space Logic

This project explores the geometric structure of human language using Linear Algebra and NLP. By representing words as 300-dimensional vectors, we can solve analogies using vector math.

## The Core Logic
In this semantic space, the relationship between words is a constant vector. I used the following formula:
Target = (Vector_B - Vector_A) + Vector_C

Example: (KING - MAN) + WOMAN = QUEEN

## Tech Stack and Tools
- Python 3.12: Environment managed via pyenv.
- spaCy (en_core_web_lg): High-resolution word vectors (300D).
- Scikit-Learn: PCA (Principal Component Analysis) to squash 300D to 2D.
- Matplotlib: To visualize the relationship map.

## Results: Parallelism
As seen in Figure_1.png, the arrows (vectors) connecting related pairs are parallel. This proves the AI has captured a consistent logical direction for concepts like gender, royalty, and tense.

## Challenges Overcome
- Dimension Mismatch: Fixed Rank-1 vs Rank-2 tensor issues using .reshape(1, -1).
- Environment Setup: Solved Silicon Mac compatibility by using Python 3.12.
- Vector Density: Upgraded from md to lg models to eliminate noisy results.

