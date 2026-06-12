# Language Modeling using Causal Transformer Decoder

Causal language models predict the probability of a word based on the preceding words in the sentence. This differs from bidirectional models, which consider both previous and following context. Here, we use a Transformer-based decoder, leveraging its attention mechanism to focus only on earlier parts of the sequence to predict the next word. This type of modeling is suitable for tasks such as text generation where the sequence order is crucial.
