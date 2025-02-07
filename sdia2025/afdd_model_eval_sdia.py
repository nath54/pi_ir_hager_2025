import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from afdd_gen3_sdia import DeepArcNet, CustomDataset_onRAM_dmp
import numpy as np  # type: ignore



#Params
pos_encode: int = 4
features_after_conv: int = 17
n_embd: int = features_after_conv*pos_encode
dropout: float = 0.0
learning_rate: float = 0.006
num_outputs: int = 6
BATCH_SIZE: int = 40




def load_trained_model(model_name, norm_coeff):
    model =  DeepArcNet()
    model.load_state_dict(torch.load(model_name, weights_only=True))
    return model, torch.load(norm_coeff)


model_name: str = "model.pth"
norm_coeff: str = "norm_coeff.tch"


model, norm_coeff = load_trained_model(model_name,norm_coeff )



trainset = CustomDataset_onRAM_dmp('./database/')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)



for batch, label in trainloader:

    prob,loss = model(batch.squeeze(0),label.squeeze(0))
    np.set_printoptions(precision=3)
    print('label', label.numpy())
    print('prob',prob.detach().numpy())




