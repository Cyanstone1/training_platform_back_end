import torch.nn as nn
import torch


class AddNorm(nn.Module):
    """
    Transformer架构里面的Add & Norm layer.
    """

    def __init__(self, normalized_shape, dropout, eps=1e-5):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """
    FFN
    """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    """

    def __init__(self, embed_dim, num_heads, ffn_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.ffn = PositionWiseFFN(embed_dim, ffn_hidden, embed_dim)
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X, need_weights=False)[0])
        return self.addnorm2(Y, self.ffn(Y))


class semi_bert(nn.Module):
    def __init__(self, vocab_size, embed_size, ffn_hiddens, num_heads, num_blks, dropout, max_len=80, **kwargs):
        super(semi_bert, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      embed_size) * 0.01)
        self.blks = nn.Sequential()

        for i in range(num_blks):
            self.blks.add_module(f"{i}", TransformerBlock(
                embed_dim=embed_size, num_heads=num_heads, ffn_hidden=ffn_hiddens, dropout=dropout))
        self.output = nn.Linear(embed_size, 2)

    def forward(self, tokens):

        # X的shape：(batch size, max_length,num_hiddens)
        X = self.token_embedding(tokens) + self.pos_embedding
        for blk in self.blks:
            X = blk(X)
            #获取句子的平均表示，而不是提取第一个字符
        X = self.output(torch.mean(X, dim=1))
        return X
