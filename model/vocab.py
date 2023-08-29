import os
from typing import Dict, List

class M2EVocab:

    PAD_IDX = 0
    
    def __init__(self, nline) -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.nline=nline

        with open(f'./dataset/dic.txt','r',encoding='utf8')as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)
        self.add_prompt()

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        print(f"Init vocab with size: {len(self.word2idx)}")

    def add_prompt(self):
        addlist=['0:sos','0:eos','1:sos','1:eos',
                 '2:sos','3:sos',]
        
        for i in range(self.nline):

            addlist.append(f'2:mol_{i}')
            addlist.append(f'2:mor_{i}')
            addlist.append(f'2:eos_{i}')

            addlist.append(f'3:mol_{i}')
            addlist.append(f'3:mor_{i}')
            addlist.append(f'3:eos_{i}')

            addlist.append(f'4:sos_{i}')
            addlist.append(f'4:mol_{i}')
            addlist.append(f'4:mor_{i}')
            addlist.append(f'4:eos_{i}')

            addlist.append(f'5:sos_{i}')
            addlist.append(f'5:mol_{i}')
            addlist.append(f'5:mor_{i}')
            addlist.append(f'5:eos_{i}')

            addlist.append(f'6:sos_{i}')
            addlist.append(f'6:mol_{i}')
            addlist.append(f'6:mor_{i}')
            addlist.append(f'6:eos_{i}')

        for w in addlist:
            self.word2idx[w] = len(self.word2idx)


    def words2indices(self, words: List[str]) -> List[int]:
        ans=[]
        for w in words:
            try:
                ans.append(self.word2idx[w])
            except KeyError:
                ans.append(self.word2idx['①']) #last word as oov
        return ans

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)
    
    def lindices2llabel(self, id_lines: List[List[int]]) -> str:
        str=''
        for line in id_lines:
            if line==[]: continue
            str+=self.indices2label(line)+' \\n '
        return str[:-4]

    def __len__(self):
        return len(self.word2idx)
    
vocab=M2EVocab(nline=16)
vocab_size=len(vocab)