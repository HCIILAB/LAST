from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from .transformer import TransformerDecoderLayer,TransformerDecoder

from .vocab import vocab, vocab_size
from .pos_enc import WordPosEnc

from .line_pos_enc import LineIntroducer, InLinePos
from .make_mask import glue_line_triu
from .datamodule import Task_batchs, BiMultinline, Plainline
import time

def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> TransformerDecoder:

    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder

class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        nline: int
    ):
        super().__init__()
        self.nline=nline
        self.word_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model))

        # self.pos_enc = WordPosEnc(d_model=d_model)
        
        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, vocab_size)
        self.d_model=d_model

        
        self.line_pos=LineIntroducer(d_model)

    def forward(self, src: FloatTensor, src_mask: LongTensor, task_seq):
        
        tgt_key_padding_mask=task_seq.input==0
        tgt=task_seq.input
        tgte = self.word_embed(tgt)
        inline_pos=task_seq.pos

        if not task_seq.task_name.startswith('plain'):
            outline_pos = self.line_pos(task_seq.li)
            tgte+=outline_pos+inline_pos
        else:
            tgte+=inline_pos

        out = self.model(
            tgt=tgte,
            memory=src,
            tgt_mask=task_seq.attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask,
        )#b l d

        out = self.proj(out)# b l d -> b l v 

        return out

    def ar(self, src: FloatTensor, src_mask: LongTensor, task_name):
        AR=Plainline(task_name, src.device, self.d_model)
        ti=0
        n=0
        while not AR.is_done():
            AR.make_input()
            st=time.perf_counter()
            new_char_outputs=self.forward(src, src_mask, AR.batch)
            et=time.perf_counter()
            ti+=et-st
            AR.update(new_char_outputs)
            n+=1
        ans=AR.return_ans()
        return ans, ti, n

    def sar(self, src: FloatTensor, src_mask: LongTensor, task_name):
        pass

    def mar(self, src: FloatTensor, src_mask: LongTensor, task_name):
        pass

    def bar(self, src: FloatTensor, src_mask: LongTensor, task_name):
        BAR=BiMultinline(task_name, src.device, self.d_model, self.nline)
        ti=0
        n=0
        while not BAR.is_done():
            BAR.make_input()
            st=time.perf_counter()
            new_char_outputs=self.forward(src, src_mask, BAR.batch)
            et=time.perf_counter()
            ti+=et-st
            BAR.update(new_char_outputs)
            n+=1
        ans=BAR.return_ans()
        return ans, ti, n










        