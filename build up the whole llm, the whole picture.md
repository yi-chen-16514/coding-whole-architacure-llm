In previous sections, we have seen the core of llm which is the multi-head attention layer. Just like a human can't only have a brain, we need hands, feet and many other organs to assembly a whole body.
This is true for llm, it is not enough to have only multi-head attention layer to be powerful, it also need other layers to work together to complete its job, first let's have a whole picture of the 
complete structure of llm like chatgpt:


<img width="1876" height="8396" alt="whiteboard_exported_image (1)" src="https://github.com/user-attachments/assets/aabb2c1e-0af2-41b0-be07-3b5e5e62e0ab" />

As we can see from above image, the input will come from the bottom, then it will go through all kinds of layers and finally output the result at the top which is predicting the next word "you". Now Let's dive deep into each layer and see how to implement them. First we setup the model parameter configuration object as following:

```py
GPT_MODEL_CONFIG = {
    "vocab_size": 502576, #total words count
    "context_length": 1024, #vector length outputed from attention layer
    "emb_dim": 768, #word vector length
    "n_heads": 12, #number of heads for attention layer,
    "n_layers": 12, #number of transformer blocks
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

First we will going to setup the skeleton, then we will fill in each block and make the skeleton with freshes:
```py
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
  def forward(self, x):
    return x

class LayerNormalization(nn.Module):
  def __init__(self, normalized_shape, eps=1e-5):
    super().__init__()

  def forward(self, x):
    return x

class GPTModelSkeleton(nn.Module):
  def __init__(sef, cfg):
    super().__init__()
    #we have 50257 words, and each word map to a vector of size 768
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length", cfg["emb_dim"]])
    self.drop_emb = nn.Dropout(cfg["drop_rate"]) #radomly set 10% of parameters in the layer to 0
    self.transformer_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = LayerNormalization(cfg["emb_dim"])
    #the final layer give out vecotr of length of words in vocab, each element represent the probability of the word as the predicted word
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)
  def forward(self, in_index):
    #batch_size is the number of sentences
    #seq_len is the max word count of each sentence
    batch_size, seq_len = in_index.shape
    tok_embeds = self.tok_emb(in_index)
    pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_index.device)
    )
```
