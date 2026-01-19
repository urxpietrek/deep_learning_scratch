import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# ---- CONFIG -----
block_size = 256
batch_size = 64
nfact = 0.9
max_iters = 5000 #50_000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
val_interval = 500 #5_000
emb_size = 384
head_size = 6
n_layers = 3
dropout = 0.2
# ---- DATA PREPARING -----
#curl -o input.txt https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt
with open('./input.txt') as file:
    text = file.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s] #change string into list of integers
decode = lambda li: ''.join([itos[i] for i in li]) #change list of integers into string

#change the whole file into list of integers
data = torch.tensor(encode(text), dtype=torch.long)

n = int(nfact*(len(data)))
xtrain = data[:n]
xval = data[n:]

torch.manual_seed(1337) #reproducability
def split_batch(split):
    data = xtrain if split == 'train' else xval
    ix = torch.randint(0, len(data)-block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losess = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = split_batch(split)
            logits, loss = model(xb, yb)
            losess[i] = loss.item()
        out[split] = losess.mean()
    model.train()
    return out
            
# ---- NAIVE BIGRAM MODEL -----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, 32) -> (B, T, head_size)
        q = self.query(x) # (B, T, 32) -> (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 #affinities (B, T, T) token:token
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        v = self.value(x) #(B, T, head_size)
        out = wei @ v #  #(B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads: int = 4):
        super().__init__()
        self.heads = [Head(head_size) for i in range(n_heads)]
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out) 
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        #Attentiom phase
        self.sa_head = MultiHeadAttention(head_size, n_head) #(B, T, emb//4) -> (B,T, emb)
        #
        self.ffwd = FeedForward(n_embd) # (B, T, emb) -> (B, T, emb)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x
        

class BigramlLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        self.position_embedding_table = nn.Embedding(block_size, emb_size)
        self.blocks = nn.Sequential(
            *[Block(emb_size, n_head=4) for _ in range(n_layers)],
            nn.LayerNorm(emb_size)
        )
        self.lm_head = nn.Linear(emb_size, vocab_size) # (B, T, emb) -> (B, T, vs)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramlLanguageModel()

m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in tqdm(range(max_iters)):
    
    if i % val_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, valid loss {losses['val']:.4f}')
    
    xb, yb = split_batch('train')
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))