import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadedAttention(nn.Module):
    """
    MultiHeadedAttention
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Take in model size and number of heads.
        :param h: number of heads
        :param d_model: embeddings dimensionality
        :param dropout: dropout rate
        """

        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0 ,'d_model must be divied by head number'
        # We assume d_v always equals d_k

        self.d_k = d_model // nhead
        self.h = nhead
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        # self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None, flag=False):
        """
        Implements Figure 2
        :param query: Query Matrix
        :param key: Key Matrix
        :param value: Value Matrix
        :param mask: Mask used during decoding
        :return:
        """

        if attn_mask is not None:
            assert attn_mask.dim()==3, f'attn_mask must be 3D, not {attn_mask.size()}. That is every batch has different mask'
            attn_mask=attn_mask[:,None,:,:]


        if key_padding_mask is not None:
            # Same mask applied to all h heads.
            assert key_padding_mask.dim()==2, f'No Need to broadcast key_padding_mask maunully, just input 2D, not {key_padding_mask.size()}'
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)#broadcast this way, must make sure is down tri

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
 
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)#make head ahead, like batch
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # query = key = value = x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) #sharing para


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, attn_mask=None, key_padding_mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention'
        :param query: Matrix with Query Embeddings [N, V, d_model]
        :param key: Matrix with Key Embeddings [N, V, d_model]
        :param value: Matrix with Value Embeddings [N, V, d_model]
        :param mask: mask used during decoding to hide future embeddings
        :param dropout: dropout value
        :return:
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, 'accept Bool only, True to mask'
            scores = scores.masked_fill(attn_mask, float('-inf'))


        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool, 'accept Bool only, True to mask'
            scores = scores.masked_fill(key_padding_mask, float('-inf'))


        p_attn = F.softmax(scores, dim=-1)
        p_attn = torch.where(torch.isnan(p_attn), torch.full_like(p_attn,0.), p_attn)# mask the nan caused by padding 

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn#b h l d/h