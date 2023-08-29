import torch

from .encoder import Encoder
from .decoder import Decoder
import time

class LAST(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        nline: int
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nline=nline
        )

    def forward(self, imgs, img_mask, task_seq):

        feature, f_mask = self.encoder(imgs, img_mask)
        out = self.decoder(feature, f_mask, task_seq)
        return out

    def ar(self, img, img_mask, task_name):
        st=time.perf_counter()
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        et=time.perf_counter()
        ans, t, n = self.decoder.ar(feature, mask, task_name)
        return ans, t+et-st, n

    def bar(self, img, img_mask, task_name):
        st=time.perf_counter()
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        et=time.perf_counter()
        ans, t, n = self.decoder.bar(feature, mask, task_name)
        return ans, t+et-st, n


if __name__=="__main__":
    model=LAST(d_model=256,
        growth_rate=24,
        num_layers=16,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.3)
    import pdb; pdb.set_trace()