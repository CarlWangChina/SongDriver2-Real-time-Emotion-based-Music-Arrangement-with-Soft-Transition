import torch
from models.emo_model import VAModel
from torch import nn


VALID_EMO_TYPE = ['median', 'concat', 'replace', 'none']
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  # param1:词嵌入字典大小； param2：每个词嵌入单词的大小
    # 正态分布初始化；e.g.,torch.nn.init.normal_(tensor, mean=0, std=1) 使值服从正态分布N(mean, std)，默认值为0，1
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


# Student Network带情感emo_type决定情感方式（中值，拼接，替代）
class SDEmbedding(nn.Module):
    def __init__(self, vocab_size, d_embed=512, d_emo=512):
        super().__init__()
        # token embedding
        self.token_src_embed = Embedding(vocab_size, d_embed)
        self.out = nn.Linear(d_embed + d_emo, d_embed)

    def forward(self, tokens, emotions):
        tokens = tokens.long()
        token_emb = self.token_src_embed(tokens)
        seq_len = tokens.shape[1]
        # 直接将知识embed和情感embed融合
        seq_emotions = emotions.unsqueeze(1)
        seq_emotions = torch.tile(seq_emotions, (1, seq_len, 1))
        # IEEE Access 拼接方法
        token_emb = torch.cat((token_emb, seq_emotions), dim=-1)
        return self.out(token_emb)



