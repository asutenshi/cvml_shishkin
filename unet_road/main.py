import unet_road
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path('roads')
    
    ds = unet_road.RoadDataset(path)

    model = unet_road.UNet().to(device)
    model.load_state_dict(torch.load('model200.pth'))
    
    img, mask = ds[0]
    img, mask = img.to(device), mask.to(device)

    result = model(img.unsqueeze(0))

    mask = mask.detach().squeeze().cpu().numpy()
    result = result.detach().squeeze().cpu().numpy()
    result[result > 0] = 1
    result[result <= 0] = 0
    

    plt.subplot(131)
    plt.imshow(mask)
    plt.title('orig')
    plt.subplot(132)
    plt.imshow(result)
    plt.title('result')
    plt.subplot(133)
    plt.imshow(mask - result)
    plt.title('diff')
    
    plt.savefig('compare.png')
    plt.show()