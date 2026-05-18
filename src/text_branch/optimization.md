### 1. Hardcore Preprocessing (Garbage In, Garbage Out)

Anime synopses pulled from APIs like AniList are notoriously messy. They contain promotional text, streaming credits, and formatting that add noise to your vectors.

* **Strip Marketing Fluff:** Use regex or a lightweight parser to remove phrases like *"Streaming on Crunchyroll!"*, *"Based on the hit manga by..."*, or *"Blu-ray release includes extra scenes."* These phrases skew semantic similarity toward licensing companies rather than actual plot elements.
* **Entity Masking:** If two different anime are both about "a high school boy who gains superpowers," but one is named Taro and the other is named Kenji, the names create statistical variance where you don’t want it. You can use an NLP library (like `spaCy`) to mask names and places into generic tokens (e.g., `[PERSON]`, `[LOCATION]`). This forces the encoder to focus strictly on the *narrative structure* rather than unique proper nouns.

### 2. Hybrid Embedding Fusion (Dense + Sparse)

Even with `e5-base`, dense embeddings struggle with exact keyword matching. They know that "magic" and "wizard" are related, but they might miss the strict statistical weight of a specific niche tag.

* **Implement BM25 / TF-IDF Parallelism:** Run a traditional sparse text encoder (like BM25) alongside your dense neural network encoder.
* **Concatenate:** Normalize the sparse scores and concatenate them to your dense vectors. This gives your downstream fusion layer the best of both worlds: the deep semantic understanding of `e5-base` combined with the rigid, exact-match keyword weight of traditional text mining.

### 3. Layer-Wise Rate Tuning (Fine-Tuning Strategy)

If you are currently using these models purely "off the shelf" (frozen feature extraction), the embeddings are still tuned for general English text, not anime-specific vocabulary (like *Isekai*, *Shonen*, *Mecha*).

* **Unfreeze the Top Layers:** Instead of keeping the entire encoder frozen, unfreeze the last 2 or 3 transformer layers.
* **Discriminative Learning Rates:** Train those top layers on your specific target dataset using a very small learning rate (e.g., $1 \times 10^{-5}$), while keeping the lower embedding layers frozen. This gently warps the model’s semantic space to better understand anime genre nuances without destroying its pre-trained English capabilities.

### 4. Dimensionality Reduction Alignment

Since you are testing larger models, `e5-base` outputs 768 dimensions (double the size of MiniLM's 384). If this is overwhelming your vision network, don't rely on the model's pooling layer alone.

* **Train a Linear Bottleneck Projection:** Add a learnable Linear Layer (`nn.Linear(768, 384)`) immediately after your text encoder output. Let the backpropagation of your entire multimodal network optimize how that 768 dimension space is compressed down to fit your vision data. This is far more effective than forcing a smaller pre-trained model to do the work.