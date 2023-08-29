import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import vocab
from bttr.model.bttr import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss, kl_loss, to_uno_tgt_out
import time, datetime
import random
class LitBTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        # training
        learning_rate: float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bttr = BTTR(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.exprate_recorder = ExpRateRecorder()
        self.time=0.
        self.n=0

    def forward(self, imgs, img_mask, task_batches) -> FloatTensor:
        return self.bttr(imgs, img_mask, task_batches)

    def training_step(self, batch, _):

        # tasks=['plain_l2r','sline_l2r','mline_l2r','bline']
        # task=random.choice(tasks)
        # task='mline_r2l'
        from .model.debug import visualize_tensor
        # visualize_tensor(batch['imgs'][0],block=False)
        
        left_hat = self(batch['imgs'], batch['img_mask'], batch['task_batches']['plain_l2r'])
        right_hat = self(batch['imgs'], batch['img_mask'], batch['task_batches']['plain_r2l'])
        bi_hat = self(batch['imgs'], batch['img_mask'], batch['task_batches']['bline'])

        left_ce = ce_loss(left_hat, batch['task_batches']['plain_l2r'].tgt)
        right_ce = ce_loss(right_hat, batch['task_batches']['plain_r2l'].tgt)
        bi_ce = ce_loss(bi_hat, batch['task_batches']['bline'].tgt)

        # triple_kl = kl_loss(left_hat, right_hat, bi_hat, batch['task_batches']['plain_l2r'].dest, batch['task_batches']['plain_r2l'].dest, batch['task_batches']['bline'].dest)
        ce_losses=left_ce+right_ce+bi_ce
        # print('CE_loss:',ce_losses.item(),'KL_loss:',triple_kl.item())
        # loss=triple_kl*100+ce_losses
        loss = ce_losses
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        task='bline'
        if task.startswith('plain'): regressor=self.bttr.ar
        if task.startswith('sline'): regressor=self.bttr.sar
        if task.startswith('mline'): regressor=self.bttr.mar
        if task.startswith('bline'): regressor=self.bttr.bar

        ans, ti, n = regressor(batch['imgs'], batch['img_mask'], task_name=task)#, task_seq=batch['task_batches'][task])

        ref = batch['task_batches'][task].gts[0]
        # self.log(
        #     "val_loss",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.exprate_recorder(ans, ref)

        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, _):
        task='bline'

        if task.startswith('plain'): regressor=self.bttr.ar
        if task.startswith('sline'): regressor=self.bttr.sar
        if task.startswith('mline'): regressor=self.bttr.mar
        if task.startswith('bline'): regressor=self.bttr.bar

        # print(vocab.indices2words(batch['task_batches'][task].tgt[0].clone().cpu().numpy()))
        # print(batch['task_batches'][task].gts[0])
        ref = batch['task_batches'][task].gts[0]
        ans, ti, n = regressor(batch['imgs'], batch['img_mask'], task_name=task)#, task_seq=batch['task_batches'][task])

        self.time+=ti
        self.n+=n
        # from .debug import visualize_tensor
        # print(ans)
        # import pdb; pdb.set_trace()
        
        # self.exprate_recorder(ans, ref)
        # import pdb; pdb.set_trace()

        # out_str=f"{batch['fnames'][0]}\t{vocab.lindices2llabel(ans)}\t{vocab.indices2label(batch['task_batches']['plain_l2r'].gts[0])}\n"
        out_str=f"{batch['fnames'][0]}\t{vocab.lindices2llabel(ans)}\t{vocab.indices2label(batch['task_batches']['plain_l2r'].gts[0])}\n"
        # out_str=''
        return out_str, ans!=ref, batch['fnames'][0]

    def test_epoch_end(self, test_outputs) -> None:
        exp_rate,e1,e2,e3,cer,score = self.exprate_recorder.compute()
        print(f"ExpRate: {exp_rate}")
        print(f"E1: {e1}")
        print(f"E2: {e2}")
        print(f"E3: {e3}")
        print(f"CER: {cer}")
        print(f"SCORE: {score}")
        print(f'NIter: {self.n}')
        print(f"Time: {self.time*1000/len(test_outputs)} ms")
        print(f"length of total file: {len(test_outputs)}")
        
        with open('test_log.txt','a',encoding='utf-8') as f:
            s=f"{str(datetime.datetime.now())}\nExpRate: {exp_rate}\nE1: {e1}\n2: {e2}\nE3: {e3}\nCER: {cer}\nSCORE: {score}\nTime: {self.time}\n"
            f.write('*'*16)
            f.write(s)

        with open('result-re.txt','w',encoding='utf8')as f:
            for out_str, nok, fname in test_outputs:
                f.write(f'{out_str}')

        # with open('not_ok-bar-on.txt','w',encoding='utf8') as f:
        #     for out_str,nok,name in test_outputs:
        #         if nok: 
        #             f.write(f'{name}\n')

        # with zipfile.ZipFile("bad_result.zip", "w") as zip_f:
        #     for img_base, pred, ok in test_outputs:
        #         if ok: continue
        #         content = f"{pred}".encode()
        #         with zip_f.open(f"{img_base}.txt", "w") as f:
        #             f.write(content)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
            verbose=True
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
