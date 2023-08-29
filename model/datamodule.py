import torch
from torch.utils.data import Dataset
import json
import cv2
import imutils
from .vocab import vocab
from .line_pos_enc import InLinePos
from .make_mask import glue_line_triu

class M2EData(Dataset):
    def __init__(self, phase):
        super(M2EData, self).__init__()
        self.list=[]
        self.dataRoot='./dataset'
        with open(self.dataRoot+f'/{phase}.jsonl','r',encoding='utf8')as f:
            for line in f:
                json_object = json.loads(line)
                self.list.append((json_object['name'],json_object['tex']))
        
    def __getitem__(self, index):
        name, tex=self.list[index]
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

        return name, img

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
    
class BiMultinline():
    def __init__(self, task_name, device, d_model, nline):
        super(BiMultinline, self).__init__()
        self.nlines = nline
        self.device = device
        self.d_model = d_model

        self.lines = [[vocab.word2idx[f'6:sos_{i}']] for i in range(self.nlines)]
        self.lines_r = [[vocab.word2idx[f'6:eos_{i}']] for i in range(self.nlines)]
        self.line_lens =[2]*self.nlines

        self.done = [False]*self.nlines
        self.pts=[]
        self.batch=Task_batchs(task_name=task_name, bz=1, max_len=256).to(device)
        self.poser = InLinePos(256)
        self.max_len=256

    def make_input(self):
        pt=0
        for i in range(self.nlines):
            this_len=self.line_lens[i]
            if pt+this_len>self.max_len-1: 
                self.done[i]=True
                return
            self.batch.input[0][pt:pt+this_len]=torch.tensor(self.lines[i]+self.lines_r[i])
            self.batch.li[0][pt:pt+this_len]=i
            pt+=this_len
            self.pts.append(pt-1)

        self.batch.attn_mask[0]=glue_line_triu(self.line_lens, bidir=True, vis_other=True, padding=self.max_len)
        self.batch.pos[0,:sum(self.line_lens)]=self.poser.get_a_poser(self.line_lens, task_name=self.batch.task_name)

    def update(self, new_char_outputs): # Such dynamic list conversion is prototype. Consider using a pre defined two dim dual-end structure in deployment.
        out_chars = new_char_outputs[0].argmax(-1)

        index = [i-j//2 for i,j in zip(self.pts, self.line_lens)]
        index_r = [i+1 for i in index]

        these_chars = out_chars[index]
        these_chars_r = out_chars[index_r]

        self.pts=[]

        for i in range(self.nlines):
            if self.done[i]: continue

            this_char = these_chars[i]
            this_char_r = these_chars_r[i]

            if 'mol' in vocab.idx2word[this_char.item()] and 'mor' in vocab.idx2word[this_char_r.item()]:
                self.done[i]=True
                continue

            self.lines[i].append(this_char.item())
            self.lines_r[i]=[this_char_r.item()]+self.lines_r[i]
            self.line_lens[i]+=2

    def is_done(self):
        if sum(self.done)==self.nlines: return True
        if sum([len(line) for line in self.lines])>=(self.max_len-self.nlines*2)//2: return True
        return  False

    def return_ans(self):
        ans=[]
        for i in range(self.nlines):
            this_line=[]
            for char in self.lines[i]+self.lines_r[i]:
                if 'sos' not in vocab.idx2word[char] and 'eos' not in vocab.idx2word[char] and 'mor' not in vocab.idx2word[char] and 'mol' not in vocab.idx2word[char]:
                    this_line.append(char)
            ans.append(this_line)
        return ans

    def print(self):
        for i in range(self.nlines):
            print(self.lines[i],' + ',self.lines_r[i])

class Plainline():
    def __init__(self, task_name, device, d_model):
        super(Plainline, self).__init__()

        self.device = device
        self.d_model = d_model
        if task_name == 'plain_l2r': starter=vocab.word2idx['0:sos']
        if task_name == 'plain_r2l': starter=vocab.word2idx['1:sos']

        self.poser=InLinePos(256)
        self.lines = [starter]
        self.done = False
        self.batch=Task_batchs(task_name=task_name, bz=1, max_len=256).to(device)

    def make_input(self):
        this_len=len(self.lines)
        if this_len>255: return self.batch
        self.batch.input[0][:this_len]=torch.tensor(self.lines).to(self.device)
        self.batch.attn_mask[0]=glue_line_triu([this_len],padding=256).to(self.device)
        self.batch.pos[0,:sum([len(self.lines)])]=self.poser.get_a_poser([len(self.lines)],task_name=self.batch.task_name)

    def update(self, new_char_outputs):
        out_chars=new_char_outputs[0].argmax(-1)
        this_char=out_chars[len(self.lines)-1]
        if this_char.item()!=vocab.word2idx['0:eos'] and this_char!=vocab.word2idx['1:eos']:
            self.lines.append(this_char.item())
        else:
            self.done=True
            
    def is_done(self):
        if self.done: return True
        if len(self.lines)>=256-2: return True
        return  False

    def return_ans(self):
        ans=[]
        for char in self.lines[1:]:
            if 'sos' not in vocab.idx2word[char] and 'eos' not in vocab.idx2word[char]:
                ans.append(char)
        return ans

    def print(self):
        print(self.lines)