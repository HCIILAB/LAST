import torch


def make_line_bitriu(out_len, tgt_len, reverse):
    assert out_len%2==0 and tgt_len%2==0, 'input len must be even'
    line_bi_whole_mask = torch.full((out_len, tgt_len), fill_value=1, dtype=torch.bool)
    half=min(out_len//2, tgt_len//2)
    line_corner_mask0 = torch.full((half, half), fill_value=1, dtype=torch.bool)
    line_corner_mask0.triu_(1)# True to mask;
    line_corner_mask1 = torch.flip(line_corner_mask0, [1])
    line_corner_mask2 = torch.flip(line_corner_mask0, [0])
    line_corner_mask3 = torch.flip(line_corner_mask0, [0,1])
    line_bi_whole_mask[:half,:half]=line_corner_mask0
    line_bi_whole_mask[:half,-half:]&=line_corner_mask1
    line_bi_whole_mask[-half:,:half]&=line_corner_mask2
    line_bi_whole_mask[-half:,-half:]&=line_corner_mask3

    if out_len>tgt_len:
        line_bi_whole_mask[tgt_len//2:out_len-tgt_len//2]=0

    # visualize_tensor(line_bi_whole_mask)

    return line_bi_whole_mask

def make_line_triu(out_len, tgt_len, reverse):
    if out_len==tgt_len:
        line_mask = torch.full((out_len, out_len), fill_value=1, dtype=torch.bool)
        line_mask.triu_(1)# True to mask
    if out_len<tgt_len:
        line_mask = torch.full((out_len, out_len), fill_value=1, dtype=torch.bool)
        line_mask.triu_(1)# True to mask
        app_mask = torch.full((out_len, tgt_len-out_len), fill_value=1, dtype=torch.bool)
        line_mask=torch.cat((line_mask,app_mask),dim=1)
    if out_len>tgt_len:
        line_mask = torch.full((tgt_len, tgt_len), fill_value=1, dtype=torch.bool)
        line_mask.triu_(1)# True to mask
        app_mask = torch.full((out_len-tgt_len,tgt_len), fill_value=0, dtype=torch.bool)
        line_mask=torch.cat((line_mask,app_mask),dim=0)
    return torch.flip(line_mask, [0,1]) if reverse else line_mask

def glue_line_triu(line_lens, reverse=False, bidir=False, vis_other=True, padding=False):
    assert not (bidir and reverse),'Could not reverse bidir in the same time'
    whole_mask=None
    func=make_line_bitriu if bidir else make_line_triu

    for i, out_len in enumerate(line_lens):
        this_out_mask=None
        for j, tgt_len in enumerate(line_lens):

            if vis_other or i==j:
                app_mask=func(out_len,tgt_len,reverse)
            else:
                app_mask=torch.full((out_len, tgt_len), fill_value=1, dtype=torch.bool)
            
            if this_out_mask is not None:
                this_out_mask=torch.cat((this_out_mask,app_mask),dim=1) 
            else:
                this_out_mask=app_mask
        if whole_mask is not None:
            whole_mask=torch.cat((whole_mask,this_out_mask),dim=0) 
        else:
            whole_mask=this_out_mask

    if padding:
        full_mask = torch.ones((padding,padding), dtype=torch.bool)
        tl=whole_mask.shape[0]
        full_mask[:tl,:tl]=whole_mask
        return full_mask

    return whole_mask

if __name__ == '__main__':
    visualize_tensor(glue_line_triu([12, 30],bidir=True,vis_other=True,reverse=False))