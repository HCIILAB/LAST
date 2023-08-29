import torch
from model.datamodule import M2EData
from lighter import Lighter
from torchvision.transforms import transforms
from eval_offline import cal_metric

setting={'d_model':256,
         'growth_rate':24,
         'num_layers':16,
         'nhead':8,
         'num_decoder_layers':3,
         'dim_feedforward':1024, 
         'dropout':0.3,
         'nline':16,
         'mode':'bline', # 'plain_l2r'
}

data_set=M2EData('test')
engine=Lighter(**setting)

model_path = 'best.pth'
dc=torch.load(model_path)

engine.load_state_dict(dc)
engine.cuda()
engine.eval()
with torch.no_grad():
    for i, batch in enumerate(data_set):
        name,img=batch
        print(f'{i}/{len(data_set)}\t{name}')
        img = transforms.ToTensor()(img)
        x_mask = torch.zeros((img.shape[1],img.shape[2]), dtype=torch.bool)
        batched={'names':[name],'imgs':img.unsqueeze(0).cuda(),'img_mask':x_mask.unsqueeze(0).cuda()}
        engine.test_step(batched)

engine.test_epoch_end()
cal_metric()
print('Time:',engine.time/len(data_set)*1000,'ms')
