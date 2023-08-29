import math
from typing import Optional

import torch
import torch.nn as nn

class InLinePos(nn.Module):
    def __init__(
        self, d_model, max_len: int = 500, temperature: float = 10000.0
    ) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pe_r = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))

        inv_freq = torch.einsum("i, j -> i j", position, div_term)
        
        pe[:, ::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()

        pe_r[:, ::2] = inv_freq.cos()
        pe_r[:, 1::2] = inv_freq.sin()
        # pe_r = pe_r.flip([0])
        self.register_buffer("pe", pe)
        self.register_buffer("pe_r", pe_r)
        self.d_model=d_model
        # from .debug import visualize_tensor
        # import pdb; pdb.set_trace()

    def get_a_poser(self, line_lens, task_name) -> torch.Tensor:
        poser=torch.zeros((sum(line_lens), self.d_model))
        pt=0
        for n in line_lens:
            if task_name.endswith('_l2r'):
                poser[pt:pt+n]=self.pe[:n]
            if task_name.endswith('_r2l'):
                # poser[pt:pt+n]=self.pe_r[-n:]
                poser[pt:pt+n]=self.pe_r[:n]
            if '_' not in task_name:
                assert n%2==0 ,'unexpected odd number when posing'
                poser[pt:pt+n//2]=self.pe[:n//2]
                # poser[pt+n//2:pt+n]=self.pe_r[-n//2:].flip([0])
                poser[pt+n//2:pt+n]=self.pe_r[:n//2].flip([0])
                # from .debug import visualize_tensor
                # import pdb; pdb.set_trace()
            pt+=n
        return poser

class LineIntroducer(nn.Module):
    def __init__(self, d_model, max_len: int = 500, temperature: float = 10000.0) -> None:
        super().__init__()
        self.Introducer=nn.Embedding(16,d_model)

    def forward(self, placeholder) -> torch.Tensor:
       
        return self.Introducer(placeholder)

if __name__ == '__main__':
    poser=LinePosEnc(d_model=512)


# if __name__ == '__main__':
#     from torchvision.utils import save_image

#     poser=LinePosEnc(d_model = 128, temperature = 10000.0)
#     mask=torch.zeros((1,512,512),dtype=torch.bool)
#     x=torch.zeros((1,512,512,128),dtype=torch.bool)
#     from torchvision.utils import save_image
#     x=poser(x, mask)

#     img_num=0
#     for _ in range(128):
#         save_image(x[0,:,:,_],'img'+str(img_num)+'.png')
#         img_num+=1

#     import imageio

#     def create_gif(gif_name, duration=0.5):
#         frames = []
#         for i in range(128):
#             frames.append(imageio.imread("img"+str(i)+".png"))
#         imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
#         return


#     gif_name = '0.gif'
#     duration = 0.2
#     create_gif(gif_name, duration)
    

