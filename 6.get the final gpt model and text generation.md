In this last section, we will see how to assemble the final gpt model from all components we created before. Actually we have been passing the most difficult parts of GPT that is 
the attention mechanism. Now what we need to do is to assembly all those components together to get the final model as following:

<img width="545" height="766" alt="空白流程图" src="https://github.com/user-attachments/assets/f34832e5-e955-465e-93f4-52858e8bd8b2" />

As we can see from above image, the GPT model is actually repeat the transformer block several times the we will done. Now Let's check the code for doing above steps:

```py
class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    self.final_norm = LayerNormalization(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)

    pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
```
Then we can try to run above code as following:

```py
torch.manual_seed(123)
model = GPTModel(GPT_MODEL_CONFIG)

out = model(batch)
print("Input batch: \n", batch)
print(f"\nOutput shape: {out.shape}")
print(out)
```

Then we can get the following result:
```py
nput batch: 
 tensor([[29688,   423,   257,   922],
        [  548,  3621,   284,   766]])

Output shape: torch.Size([2, 4, 50257])
tensor([[[-1.0453,  0.1800, -0.0803,  ..., -1.0559, -0.9745, -0.3132],
         [-0.7219, -0.2006, -0.2538,  ..., -0.6076, -0.9438, -0.3187],
         [-0.7586,  0.1957, -0.5045,  ..., -0.1706, -0.2094, -0.1008],
         [ 0.1775, -0.2065,  0.6359,  ...,  0.4498,  0.6831,  0.5175]],

        [[-1.0495, -0.5007, -0.7872,  ..., -0.6699, -0.6705, -0.1962],
         [-0.4351, -0.5589, -0.8543,  ..., -0.5571, -0.5653,  0.3484],
         [-0.4010, -0.3221,  0.3346,  ..., -0.1267,  0.2264,  0.4347],
         [-0.4100, -1.1526,  1.4263,  ...,  0.7129, -0.1073,  1.0928]]],
```
To be noticed, the finall result is vector with elements of the vocab size, and the last vector contains probability of each word of the vocab for the next output word.
