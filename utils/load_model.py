import torch
from TranSalNet_Res import TranSalNet


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TranSalNet()
    model.load_state_dict(torch.load(
        r'pretrained_models/TranSalNet_Res.pth', map_location=torch.device('cpu')))

    model = model.to(device)
    return model
