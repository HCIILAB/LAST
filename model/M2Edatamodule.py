from torch.utils.data import Dataset
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from typing import List, Optional, Tuple
from easydict import EasyDict as edict
import cv2
import torch
from torchvision.transforms import transforms
import imutils
import numpy as np
from .vocab import ALVocab
from .make_mask import glue_line_triu
from ..model.line_pos_enc import InLinePos

import copy

vocab = ALVocab()

def two_dimensionalize(tokens):
    lines=[]
    s=0
    for i,t in enumerate(tokens):
        if t == '\\t' or t == '\\n':
            assert s<i, tokens
            lines.append(tokens[s:i])
            s=i+1
        if i == len(tokens)-1:
            assert s<i+1
            lines.append(tokens[s:i+1])
    assert tokens[-1] not in ['\\n', '\\t']
    assert len(lines)<=16
    for i in range(16-len(lines)):# for uni-stage
        lines.append([])
    return lines

def two_dimensionalize_v2(input_list):
    result_list = []
    current_sublist = []

    for item in input_list:
        if item == '\t' or item == '\n':
            if current_sublist:
                result_list.append(current_sublist)
                current_sublist = []
        else:
            current_sublist.append(item)

    if current_sublist:
        result_list.append(current_sublist)

    assert len(result_list)<=16
    for i in range(16-len(result_list)):# for 
        result_list.append([])

    return result_list

def reverse(list):# deep reverse a list
    list_re=copy.deepcopy(list)
    list_re.reverse()
    return list_re

def insert(list,index,element):# deep insert a list
    list_in=copy.deepcopy(list)
    list_in.insert(index, element)
    return list_in

def cut2(line):
    line_len=len(line)
    odd=line_len%2
    if odd:
        left=line[:line_len//2 + 1]
        right=line[line_len//2 + 1:]# mor padding
    else:
        left=line[:line_len//2]
        right=line[line_len//2:]
    return left, right, odd

class Task_seqs():
    def __init__(self,task_name):
        super(Task_seqs, self).__init__()
        self.task_name=task_name
        self.gt=[]
        self.input=[]
        self.tgt=[]
        self.li=[]
        self.lens=[]
        self.attn_mask=[]
        self.pos=None
        self.dest=[]

    def tensorize(self, inposer):
        self.Tinput=torch.LongTensor(vocab.words2indices(self.input))
        self.Ttgt=torch.LongTensor(vocab.words2indices(self.tgt))
        self.Tli=torch.LongTensor(self.li)
        self.gt=vocab.words2indices(self.gt)
        self.pos=inposer.get_a_poser(self.lens, self.task_name)
        self.Tdest=torch.LongTensor(self.dest)
        assert self.Tinput.shape[0]==self.Ttgt.shape[0], f'len mismatch while tensorizing task {self.task_name}!'

        if self.Ttgt.shape[0]!=self.Tdest.shape[0] and (self.task_name.startswith('plain') or self.task_name.startswith('bline')):
            print(f'len mismatch in plain')
            

def make_seqs(tokens, inposer):
    tokens_re=reverse(tokens)

    task_names=['plain_l2r','plain_r2l','sline_l2r','sline_r2l','mline_l2r','mline_r2l','bline']
    task_seqs=[Task_seqs(task_name) for task_name in task_names]
    task_seqs=edict(zip(task_names,task_seqs))

    task_seqs.plain_l2r.input=['0:sos']+tokens
    task_seqs.plain_l2r.tgt=tokens+['0:eos']
    task_seqs.plain_l2r.gt=tokens
    # task_seqs.plain_l2r.li=[0]*len(task_seqs.plain_l2r.input)
    task_seqs.plain_l2r.dest=[]
    task_seqs.plain_l2r.lens.append(len(task_seqs.plain_l2r.input))

    task_seqs.plain_r2l.input=['1:sos']+tokens_re
    task_seqs.plain_r2l.tgt=tokens_re+['1:eos']
    task_seqs.plain_r2l.gt=tokens_re
    # task_seqs.plain_r2l.li=[0]*len(task_seqs.plain_r2l.input)
    task_seqs.plain_r2l.dest=[]
    task_seqs.plain_r2l.lens.append(len(task_seqs.plain_r2l.input))

    task_seqs.sline_l2r.input.append('2:sos')
    task_seqs.sline_r2l.input.append('3:sos')

    task_seqs.sline_l2r.li.append(0)
    task_seqs.sline_r2l.li.append(0)

    task_seqs.bline.dest=[]

    lines=two_dimensionalize(tokens)

    for i in range(16):
        mid_pad=False
        line=lines[i]
        
        l2r_left, l2r_right, mid_pad=cut2(line)
        r2l_right, r2l_left=reverse(l2r_left), reverse(l2r_right)

        len_l=len(l2r_left)
        len_r=len(l2r_right)
        if len_l>0 or len_r>0:
            # import pdb; pdb.set_trace()
            task_seqs.plain_l2r.dest+=[i]*len(l2r_left)+[i+100]*len(l2r_right)+[-1]
            task_seqs.plain_r2l.dest+=[i+100]*len(r2l_left)+[i]*len(r2l_right)+[-1]

        if not mid_pad:
            sline_l2r_stuff=l2r_left+[f'2:mol_{i}',f'2:mor_{i}']+l2r_right+[f'2:eos_{i}']
            task_seqs.sline_l2r.input+=sline_l2r_stuff
            task_seqs.sline_l2r.tgt+=sline_l2r_stuff
            task_seqs.sline_l2r.gt+=l2r_left+l2r_right
            task_seqs.sline_l2r.li+=[i]*len(sline_l2r_stuff)
            task_seqs.sline_l2r.lens.append(len(sline_l2r_stuff))

            sline_r2l_stuff=r2l_left+[f'3:mol_{i}',f'3:mor_{i}']+r2l_right+[f'3:eos_{i}']
            task_seqs.sline_r2l.input+=sline_r2l_stuff
            task_seqs.sline_r2l.tgt+=sline_r2l_stuff
            task_seqs.sline_r2l.gt+=r2l_left+r2l_right
            task_seqs.sline_r2l.li+=[i]*len(sline_r2l_stuff)
            task_seqs.sline_r2l.lens.append(len(sline_r2l_stuff))

            task_seqs.mline_l2r.input+=[f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}']+l2r_right
            task_seqs.mline_l2r.tgt+=l2r_left+[f'4:mol_{i}',f'4:mor_{i}']+l2r_right+[f'4:eos_{i}']
            task_seqs.mline_l2r.gt+=l2r_left+l2r_right
            task_seqs.mline_l2r.li+=[i]*len([f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}']+l2r_right)
            task_seqs.mline_l2r.lens.append(len([f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}']+l2r_right))

            task_seqs.mline_r2l.input+=[f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mor_{i}']+r2l_right
            task_seqs.mline_r2l.tgt+=r2l_left+[f'5:mol_{i}',f'5:mor_{i}']+r2l_right+[f'5:eos_{i}']
            task_seqs.mline_r2l.gt+=r2l_left+r2l_right
            task_seqs.mline_r2l.li+=[i]*len([f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mor_{i}']+r2l_right)
            task_seqs.mline_r2l.lens.append(len([f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mor_{i}']+r2l_right))

            task_seqs.bline.input+=[f'6:sos_{i}']+l2r_left+l2r_right+[f'6:eos_{i}']
            task_seqs.bline.tgt+=l2r_left+[f'6:mol_{i}',f'6:mor_{i}']+l2r_right
            task_seqs.bline.gt+=l2r_left+l2r_right
            task_seqs.bline.li+=[i]*len([f'6:sos_{i}']+l2r_left+l2r_right+[f'6:eos_{i}'])
            task_seqs.bline.lens.append(len([f'6:sos_{i}']+l2r_left+l2r_right+[f'6:eos_{i}']))
            task_seqs.bline.dest+=[i]*len(l2r_left)+[-1, -1]+[i+100]*len(l2r_right)

        else:
            sline_l2r_stuff=l2r_left+[f'2:mol_{i}',f'2:mor_{i}',f'2:mor_{i}']+l2r_right+[f'2:eos_{i}']
            task_seqs.sline_l2r.input+=sline_l2r_stuff
            task_seqs.sline_l2r.tgt+=sline_l2r_stuff
            task_seqs.sline_l2r.gt+=l2r_left+l2r_right
            task_seqs.sline_l2r.li+=[i]*len(sline_l2r_stuff)
            task_seqs.sline_l2r.lens.append(len(sline_l2r_stuff))

            sline_r2l_stuff=r2l_left+[f'3:mol_{i}',f'3:mol_{i}',f'3:mor_{i}']+r2l_right+[f'3:eos_{i}']
            task_seqs.sline_r2l.input+=r2l_left+[f'3:mol_{i}',f'3:mol_{i}',f'3:mor_{i}']+r2l_right+[f'2:eos_{i}']
            task_seqs.sline_r2l.tgt+=r2l_left+[f'3:mol_{i}',f'3:mol_{i}',f'3:mor_{i}']+r2l_right+[f'2:eos_{i}']
            task_seqs.sline_r2l.gt+=r2l_left+r2l_right
            task_seqs.sline_r2l.li+=[i]*len(sline_r2l_stuff)
            task_seqs.sline_r2l.lens.append(len(sline_r2l_stuff))

            task_seqs.mline_l2r.input+=[f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}',f'4:mor_{i}']+l2r_right
            task_seqs.mline_l2r.tgt+=l2r_left+[f'4:mol_{i}',f'4:mor_{i}',f'4:mor_{i}']+l2r_right+[f'4:eos_{i}']
            task_seqs.mline_l2r.gt+=l2r_left+l2r_right
            task_seqs.mline_l2r.li+=[i]*len([f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}',f'4:mor_{i}']+l2r_right)
            task_seqs.mline_l2r.lens.append(len([f'4:sos_{i}']+l2r_left+[f'4:mol_{i}',f'4:mor_{i}',f'4:mor_{i}']+l2r_right))

            task_seqs.mline_r2l.input+=[f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mol_{i}',f'5:mor_{i}']+r2l_right
            task_seqs.mline_r2l.tgt+=r2l_left+[f'5:mol_{i}',f'5:mol_{i}',f'5:mor_{i}']+r2l_right+[f'5:eos_{i}']
            task_seqs.mline_r2l.gt+=r2l_left+r2l_right
            task_seqs.mline_r2l.li+=[i]*len([f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mol_{i}',f'5:mor_{i}']+r2l_right)
            task_seqs.mline_r2l.lens.append(len([f'5:sos_{i}']+r2l_left+[f'5:mol_{i}',f'5:mol_{i}',f'5:mor_{i}']+r2l_right))

            task_seqs.bline.input+=[f'6:sos_{i}']+l2r_left+[f'6:mor_{i}']+l2r_right+[f'6:eos_{i}']
            task_seqs.bline.tgt+=l2r_left+[f'6:mol_{i}',f'6:mor_{i}',f'6:mor_{i}']+l2r_right
            task_seqs.bline.gt+=l2r_left+l2r_right
            task_seqs.bline.li+=[i]*len([f'6:sos_{i}']+l2r_left+[f'6:mor_{i}']+l2r_right+[f'6:eos_{i}'])
            task_seqs.bline.lens.append(len([f'6:sos_{i}']+l2r_left+[f'6:mor_{i}']+l2r_right+[f'6:eos_{i}']))
            task_seqs.bline.dest+=[i]*len(l2r_left)+[-1, -1, -1]+[i+100]*len(l2r_right)
    
    task_seqs.sline_l2r.input=task_seqs.sline_l2r.input[:-1] # cut last eos in input
    task_seqs.sline_r2l.input=task_seqs.sline_r2l.input[:-1] # cut last eos in input
    task_seqs.sline_l2r.li=task_seqs.sline_l2r.li[:-1]
    task_seqs.sline_r2l.li=task_seqs.sline_r2l.li[:-1]

    task_seqs.plain_l2r.attn_mask=glue_line_triu([len(task_seqs.plain_l2r.input)], reverse=False, bidir=False, vis_other=True)
    task_seqs.plain_r2l.attn_mask=glue_line_triu([len(task_seqs.plain_r2l.input)], reverse=False, bidir=False, vis_other=True)
    task_seqs.sline_l2r.attn_mask=glue_line_triu([len(task_seqs.sline_l2r.input)], reverse=False, bidir=False, vis_other=True)
    task_seqs.sline_r2l.attn_mask=glue_line_triu([len(task_seqs.sline_r2l.input)], reverse=False, bidir=False, vis_other=True)
    task_seqs.mline_l2r.attn_mask=glue_line_triu(task_seqs.mline_l2r.lens, reverse=False, bidir=False, vis_other=True)
    task_seqs.mline_r2l.attn_mask=glue_line_triu(task_seqs.mline_r2l.lens, reverse=False, bidir=False, vis_other=True)
    task_seqs.bline.attn_mask=glue_line_triu(task_seqs.bline.lens, reverse=False, bidir=True, vis_other=True)

    for v in task_seqs.values():
        v.tensorize(inposer)

    return task_seqs
    
class ALData(Dataset):
    def __init__(self, phase):
        super(ALData, self).__init__()
        self.list=[]
        self.dataRoot='../../M2E_Dataset'
        with open(self.dataRoot+f'/{phase}.txt','r',encoding='utf8')as f:
            lines=f.readlines()
        for line in lines:
            tmp=line.strip().split('\t')
            name,tex=tmp
            # if phase=='train' and '-ex' in name: continue
            tokens=tex.replace('\\n \\t', '\\t').split(' ')
            try:
                two_dimensionalize(tokens)
            except AssertionError:
                print(f'{name} is too long, ignore')
                continue
            self.list.append((name,tokens))

        self.inposer=InLinePos(256)
        if phase == 'val': 
            self.list=self.list[:5000]
        # if phase == 'test': 
        #     self.list=self.list[:10]


    def __getitem__(self, index):
        name,tex=self.list[index]
        img = cv2.imread(self.dataRoot+f'/images/{name}', cv2.IMREAD_COLOR)
        assert img is not None
        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        top_size=(256-img.shape[0])//2
        bottom_size=256-top_size-img.shape[0]

        left_size=(256-img.shape[1])//2
        right_size=256-left_size-img.shape[1]

        img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT, value = (255,255,255))

        seqs=make_seqs(tex, self.inposer)
        
        # masks = [glue_line_triu([seqs], reverse=False, bidir=False, vis_other=False),
        #          glue_line_triu([stgt.shape[0]], reverse=True, bidir=False, vis_other=True),
        #          glue_line_triu(line_lens, reverse=False, bidir=False, vis_other=True),
        #          glue_line_triu(line_lens, reverse=False, bidir=False, vis_other=False),
        #          glue_line_triu(line_lens, reverse=True, bidir=False, vis_other=True),
        #          glue_line_triu(line_lens, reverse=True, bidir=False, vis_other=False),
        #          glue_line_triu(bi_line_lens, reverse=False, bidir=True, vis_other=True),
        #          glue_line_triu(bi_line_lens, reverse=False, bidir=True, vis_other=False)]

        # inline_pos=[self.poser.get_poser([stgt.shape[0]], mode='norm'),
        #             self.poser.get_poser([stgt.shape[0]], mode='reverse'),
        #             self.poser.get_poser(line_lens, mode='norm'),
        #             self.poser.get_poser(line_lens, mode='reverse'),
        #             self.poser.get_poser(bi_line_lens, mode='bidir')]

        return name, img, seqs

    def __len__(self):
        return len(self.list)

class Task_batchs():
    def __init__(self,task_name, bz, max_len):
        super(Task_batchs, self).__init__()
        self.task_name=task_name
        self.bz=0
        self.gts=[]
        self.input=torch.zeros((bz, max_len), dtype=torch.long)
        self.tgt=torch.zeros((bz, max_len), dtype=torch.long)
        self.pos=torch.zeros((bz, max_len, 256))
        self.li=torch.zeros((bz, max_len), dtype=torch.long)
        self.attn_mask=torch.ones((bz, max_len, max_len), dtype=torch.bool)
        self.dest=torch.full((bz, max_len), fill_value=-1, dtype=torch.long)
        self.lens=[]
    def to(self, device):
        self.input=self.input.to(device)
        self.tgt=self.tgt.to(device)
        self.attn_mask=self.attn_mask.to(device)
        self.li=self.li.to(device)
        self.pos=self.pos.to(device)
        return self

def collate(batch):
    xs=[]
    ys=[]
    plain_lens=[]
    sline_lens=[]
    mline_lens=[]
    bline_lens=[]

    bz=len(batch)
    for it in batch:
        xs.append(it[1].shape[0])
        ys.append(it[1].shape[1])
        plain_lens.append(sum(it[2].plain_l2r.lens))
        sline_lens.append(sum(it[2].sline_l2r.lens))
        mline_lens.append(sum(it[2].mline_l2r.lens))
        bline_lens.append(sum(it[2].bline.lens))

    xmax=max(xs)
    ymax=max(ys)
    plain_max_len=max(plain_lens)
    sline_max_len=max(sline_lens)
    mline_max_len=max(mline_lens)
    bline_max_len=max(bline_lens)

    x = torch.zeros(bz, 3, xmax, ymax)
    x_mask = torch.ones(bz, xmax, ymax, dtype=torch.bool)

    task_batches=edict()
    task_batches['plain_l2r']=Task_batchs('plain_l2r', bz, plain_max_len)
    task_batches['plain_r2l']=Task_batchs('plain_r2l', bz, plain_max_len)
    task_batches['sline_l2r']=Task_batchs('sline_l2r', bz, sline_max_len)
    task_batches['sline_r2l']=Task_batchs('sline_r2l', bz, sline_max_len)
    task_batches['mline_l2r']=Task_batchs('mline_l2r', bz, mline_max_len)
    task_batches['mline_r2l']=Task_batchs('mline_r2l', bz, mline_max_len)
    task_batches['bline']=Task_batchs('bline', bz, bline_max_len)

    fnames=[]
    lines_gt=[]
    num_lines=[]

    for idx in range(bz):
        fnames.append(batch[idx][0])

        x[idx,: , : xs[idx], : ys[idx]] = transforms.ToTensor()(batch[idx][1])
        x_mask[idx, : xs[idx], : ys[idx]] = False

        task_batches.plain_l2r.input[idx, :plain_lens[idx]]=batch[idx][2].plain_l2r.Tinput
        task_batches.plain_l2r.tgt[idx, :plain_lens[idx]]=batch[idx][2].plain_l2r.Ttgt
        task_batches.plain_l2r.gts.append(batch[idx][2].plain_l2r.gt)
        task_batches.plain_l2r.attn_mask[idx, :plain_lens[idx], :plain_lens[idx]]=batch[idx][2].plain_l2r.attn_mask
        task_batches.plain_l2r.lens.append(batch[idx][2].plain_l2r.lens)
        task_batches.plain_l2r.pos[idx, :plain_lens[idx]]=batch[idx][2].plain_l2r.pos
        # task_batches.plain_l2r.dest[idx, :plain_lens[idx]]=batch[idx][2].plain_l2r.Tdest

        task_batches.plain_r2l.input[idx, :plain_lens[idx]]=batch[idx][2].plain_r2l.Tinput
        task_batches.plain_r2l.tgt[idx, :plain_lens[idx]]=batch[idx][2].plain_r2l.Ttgt
        task_batches.plain_r2l.gts.append(batch[idx][2].plain_r2l.gt)
        task_batches.plain_r2l.attn_mask[idx, :plain_lens[idx], :plain_lens[idx]]=batch[idx][2].plain_r2l.attn_mask
        task_batches.plain_r2l.lens.append(batch[idx][2].plain_r2l.lens)
        task_batches.plain_r2l.pos[idx, :plain_lens[idx]]=batch[idx][2].plain_r2l.pos
        # task_batches.plain_r2l.dest[idx, :plain_lens[idx]]=batch[idx][2].plain_r2l.Tdest

        task_batches.sline_l2r.input[idx, :sline_lens[idx]]=batch[idx][2].sline_l2r.Tinput
        task_batches.sline_l2r.tgt[idx, :sline_lens[idx]]=batch[idx][2].sline_l2r.Ttgt
        task_batches.sline_l2r.li[idx, :sline_lens[idx]]=batch[idx][2].sline_l2r.Tli
        task_batches.sline_l2r.gts.append(batch[idx][2].sline_l2r.gt)
        task_batches.sline_l2r.attn_mask[idx, :sline_lens[idx], :sline_lens[idx]]=batch[idx][2].sline_l2r.attn_mask
        task_batches.sline_l2r.lens.append(batch[idx][2].sline_l2r.lens)
        task_batches.sline_l2r.pos[idx, :sline_lens[idx]]=batch[idx][2].sline_l2r.pos

        task_batches.sline_r2l.input[idx, :sline_lens[idx]]=batch[idx][2].sline_r2l.Tinput
        task_batches.sline_r2l.tgt[idx, :sline_lens[idx]]=batch[idx][2].sline_r2l.Ttgt
        task_batches.sline_r2l.li[idx, :sline_lens[idx]]=batch[idx][2].sline_r2l.Tli
        task_batches.sline_r2l.gts.append(batch[idx][2].sline_r2l.gt)
        task_batches.sline_r2l.attn_mask[idx, :sline_lens[idx], :sline_lens[idx]]=batch[idx][2].sline_r2l.attn_mask
        task_batches.sline_r2l.lens.append(batch[idx][2].sline_r2l.lens)
        task_batches.sline_r2l.pos[idx, :sline_lens[idx]]=batch[idx][2].sline_r2l.pos

        task_batches.mline_l2r.input[idx, :mline_lens[idx]]=batch[idx][2].mline_l2r.Tinput
        task_batches.mline_l2r.tgt[idx, :mline_lens[idx]]=batch[idx][2].mline_l2r.Ttgt
        task_batches.mline_l2r.li[idx, :mline_lens[idx]]=batch[idx][2].mline_l2r.Tli
        task_batches.mline_l2r.gts.append(batch[idx][2].mline_l2r.gt)
        task_batches.mline_l2r.attn_mask[idx, :mline_lens[idx], :mline_lens[idx]]=batch[idx][2].mline_l2r.attn_mask
        task_batches.mline_l2r.lens.append(batch[idx][2].mline_l2r.lens)
        task_batches.mline_l2r.pos[idx, :mline_lens[idx]]=batch[idx][2].mline_l2r.pos

        task_batches.mline_r2l.input[idx, :mline_lens[idx]]=batch[idx][2].mline_r2l.Tinput
        task_batches.mline_r2l.tgt[idx, :mline_lens[idx]]=batch[idx][2].mline_r2l.Ttgt
        task_batches.mline_r2l.li[idx, :mline_lens[idx]]=batch[idx][2].mline_r2l.Tli
        task_batches.mline_r2l.gts.append(batch[idx][2].mline_r2l.gt)
        task_batches.mline_r2l.attn_mask[idx, :mline_lens[idx], :mline_lens[idx]]=batch[idx][2].mline_r2l.attn_mask
        task_batches.mline_r2l.lens.append(batch[idx][2].mline_r2l.lens)
        task_batches.mline_r2l.pos[idx, :mline_lens[idx]]=batch[idx][2].mline_r2l.pos

        task_batches.bline.input[idx, :bline_lens[idx]]=batch[idx][2].bline.Tinput
        task_batches.bline.tgt[idx, :bline_lens[idx]]=batch[idx][2].bline.Ttgt
        task_batches.bline.li[idx, :bline_lens[idx]]=batch[idx][2].bline.Tli
        task_batches.bline.gts.append(batch[idx][2].bline.gt)
        task_batches.bline.attn_mask[idx, :bline_lens[idx], :bline_lens[idx]]=batch[idx][2].bline.attn_mask
        task_batches.bline.lens.append(batch[idx][2].bline.lens)
        task_batches.bline.pos[idx, :bline_lens[idx]]=batch[idx][2].bline.pos
        task_batches.bline.dest[idx, :bline_lens[idx]]=batch[idx][2].bline.Tdest

    return {'fnames':fnames, 'imgs':x, 'img_mask':x_mask, 'task_batches':task_batches}

class ALDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Load data from: Multi-line")

    def setup(self, stage: Optional[str] = None) -> None:
            if stage == "fit" or stage is None:
                self.train_dataset = ALData('train')
                self.val_dataset = ALData('val')
            if stage == "test" or stage is None:
                self.test_dataset = ALData('test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

# if __name__ == "__main__":
    # ptdm=ALDatamodule()
    # ptdm.setup()
    # for data in ptdm.train_dataset :
    #     print(data)
    #     exit()

    # train_dataset=ALData('train')
    # test_loader=DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=0, collate_fn=collate)
    # for batch in test_loader:
    #     print(batch)
    #     exit()