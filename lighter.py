import torch
from model.last import LAST
from model.vocab import vocab
class Lighter(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        nline: int,
        mode: str
    ):
        super().__init__()
        self.last = LAST(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nline=nline
        )
        self.mode=mode
        self.time=0.
        self.n=0
        self.outs=[]

    def forward(self, imgs, img_mask, task_batches):
        return self.last(imgs, img_mask, task_batches)

    def training_step(self, batch, _):
        pass

    def validation_step(self, batch, _):
        pass

    def test_step(self, batch):
        if self.mode.startswith('plain'): regressor=self.last.ar
        # if task.startswith('sline'): regressor=self.last.sar
        # if task.startswith('mline'): regressor=self.last.mar
        if self.mode.startswith('bline'): regressor=self.last.bar

        ans, ti, n = regressor(batch['imgs'], batch['img_mask'], task_name=self.mode)
        if batch['names'][0]=='73970.jpg':
            import pdb; pdb.set_trace()
        self.outs.append((batch['names'][0], ans, ti))

        self.time+=ti
        self.n+=n


    def test_epoch_end(self):
        with open('results.txt','w',encoding='utf8')as f:
            for entry in self.outs:
                pr=''
                if self.mode=='bline':
                    pr=vocab.lindices2llabel(entry[1])
                if self.mode.startswith('plain'):
                    pr=vocab.indices2label(entry[1])
                    
                out_str=f'{entry[0]}\t{pr}\n'
                f.write(out_str)



        
            

        

