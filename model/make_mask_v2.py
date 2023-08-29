import torch

def make_same_line(k):
    assert k % 2 == 0, "k must be an even number"
    m = torch.zeros((k, k),dtype=torch.bool)
    for p in range(1, k+1): # Note that the indices in Eq(2) are one-based.
        for q in range(1, k+1):
            if (p < k // 2 and p < q < k - p + 1) or (p > k // 2 and k - p + 1 < q < p):
                m[p-1, q-1] = 1
    return m

def make_other_line(out_len, in_len):
    assert out_len % 2 == 0, "out_len must be an even number"
    assert in_len % 2 == 0, "in_len must be an even number"

    m = torch.ones((out_len, in_len),dtype=torch.bool)
    for p in range(1, out_len+1):
        for q in range(1, in_len+1):
            if  (p <= out_len // 2 and q <= in_len // 2 and p>=q) or (p <= out_len // 2 and q > in_len // 2 and p > in_len-q) or (p > out_len // 2 and q <= in_len // 2 and out_len-p+1 >= q) or (p > out_len // 2 and q > in_len // 2 and out_len-p >= in_len-q):
                m[p-1, q-1] = 0
    return m

def where2tile(line_lens,i,j):
    s0=sum(line_lens[:i])
    e0=sum(line_lens[:i+1])
    s1=sum(line_lens[:j])
    e1=sum(line_lens[:j+1])

    return (s0, s1), (e0, e1)

def gen_line_triu(line_lens):
    whole_mask=torch.ones(sum(line_lens),sum(line_lens),dtype=torch.bool)
    for i, out_len in enumerate(line_lens):
        for j, in_len in enumerate(line_lens):
            if i==j:
                this_partition=make_same_line(out_len)
            else:
                this_partition=make_other_line(out_len, in_len)
            # import pdb; pdb.set_trace()
            si,ei=where2tile(line_lens,i,j)
            whole_mask[si[0]:ei[0],si[1]:ei[1]]=this_partition
    return whole_mask
        
# gen_line_triu([6, 4, 4],vis_other=True,reverse=False)
if __name__=='__main__':
    visualize_tensor(gen_line_triu([12, 30]))
    # visualize_tensor(make_other_line(12+10,4+10))
    # visualize_tensor(make_other_line(4))
    # visualize_tensor(make_other_line(2))