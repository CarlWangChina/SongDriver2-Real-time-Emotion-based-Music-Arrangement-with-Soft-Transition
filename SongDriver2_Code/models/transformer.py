import math
import torch
import numpy as np
import torch.nn as nn
import pickle
from models.SDEmbedding import SDEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


basetone2idx = {
    'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11
}
keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
full_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

note2token = {
    "Cbb": "Bb",
    "Cb": "B",
    "C": "C",
    "C#": "Db",
    "C##": "D",
    "Dbb": "C",
    "Db": "Db",
    "D": "D",
    "D#": "Eb",
    "D##": "E",
    "Ebb": "D",
    "Eb": "Eb",
    "E": "E",
    "E#": "F",
    "E##": "F#",
    "Fbb": "Eb",
    "Fb": "E",
    "F": "F",
    "F#": "F#",
    "F##": "G",
    "Gbb":"F",
    "Gb": "F#",
    "G": "G",
    "G#": "Ab",
    "G##": "A",
    "Abb": "G",
    "Ab": "Ab",
    "A": "A",
    "A#": "Bb",
    "A##": "B",
    "Bbb": "A",
    "Bb": "Bb",
    "B": "B",
    "B#": "C",
    "B##": "Db",
}


def normalise_key (keyname):
    key, quality = keyname.split('.')
    key = key[0].upper() + key[1:]
    key = note2token[key]
    k, q = key, quality
    q = 'Major' if quality.strip().upper() == 'MAJOR' else 'minor'
    if len(key) == 2:
        if quality.strip().upper() == 'MAJOR': #b
            if key[-1] == '#':
                k = keys[(keys.index(k[0].upper())+1)%7].upper() + 'b'
                # k = k[0].upper()+k[1] if quality.strip().upper() == 'MAJOR' else k[0].lower()+k[1]
            else:
                k = k[0].upper() + k[1]
        else: ##
            if key[-1] == 'b':
                k = keys[(keys.index(k[0].upper())-1+7)%7].lower() + '#'
            else:
                k = k[0].lower() + k[1]

    else:
        k = k.upper() if quality.strip().upper() == 'MAJOR' else k.lower()
    return k+'_'+q


def tone2idx (tonename):
    if len(tonename.strip()) > 1:
        root, shift = tonename[0], tonename[1]
    else:
        root, shift = tonename[0], ''
    index = 0
    baseC = basetone2idx['C']
    index = basetone2idx[root.upper()] - baseC
    if shift == '':
        return index
    elif shift == 'b':
        return (index - 1 + 12) % 12
    elif shift == '#':
        return (index + 1) % 12

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # [batch_size, 1, len_k], False is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # Upper triangular matrix
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = args.d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)
        self.scaledDotProductAttention = ScaledDotProductAttention(args)
        self.n_heads = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.scaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # x = nn.LayerNorm(self.d_model)(output + residual).to_device()
        return self.layernorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )
        self.d_model = args.d_model
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        residual = residual
        output = self.fc(inputs)
        output = output
        # [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_self_attn_mask = enc_self_attn_mask
        enc_inputs = enc_inputs
        self.enc_self_attn = self.enc_self_attn
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, src_len, d_model]
        attn = attn
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = enc_outputs
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.dec_enc_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.d_model = args.d_model
        self.src_emb = SDEmbedding(args.src_vocab_size, args.d_model, args.d_emb_emo)
        self.pos_emb = PositionalEncoding(args.d_model, args.dropout)
        self.layers = nn.ModuleList([EncoderLayer(args)
                                    for _ in range(args.n_layers)])

    def forward(self, enc_inputs, emotions):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs, emotions)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.d_model = args.d_model
        self.tgt_emb = SDEmbedding(args.tgt_vocab_size, args.d_model, args.d_emb_emo)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([DecoderLayer(args)
                                    for _ in range(args.n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs, emotions):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs, emotions)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(dec_outputs.device)  # [batch_size, tgt_len, tgt_len]

        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(dec_outputs.device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batch_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class DualTransformer(nn.Module):
    def __init__(self, args):
        super(DualTransformer, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.projection_notes = nn.Linear(
            args.d_model, 113, bias=False)
        self.projection_chords = nn.Linear(
            args.d_model, 17327, bias=False)
    def forward(self, enc_inputs, dec_inputs, emotions):
        '''
        enc_inputs: [batch_size, src_lens]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, emotions)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, emotions)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        logits_notes = self.projection_notes(dec_outputs)
        logits_chords = self.projection_chords(dec_outputs)[:, :17, :]
        return logits_notes.reshape(-1, logits_notes.size(-1)), logits_chords.reshape(-1, logits_chords.size(-1))


if __name__ == '__main__':
    pass
