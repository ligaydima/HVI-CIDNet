from net.CIDNet import CIDNet
import argparse
import torch
import os
from PIL import Image
from data.data import get_SICE_eval_set
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

device = torch.device("cpu")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    model = CIDNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    eval_data = DataLoader(dataset=get_SICE_eval_set(args.input), batch_size=1, shuffle=False)
    if not os.path.exists(args.output):          
        os.mkdir(args.output)
    for batch in tqdm(eval_data):
        with torch.no_grad():
            input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            input = input.to(device)
            output = model(input) 
        output = torch.clamp(output, 0, 1)
        output = output[:, :, :h, :w]
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(os.path.join(args.output, name[0]))
    
if __name__ == "__main__":
    main()
