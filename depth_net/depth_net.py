import depth_net.networks as networks
import torch
import PIL.Image
from torchvision import transforms, datasets
from depth_net.layers import disp_to_depth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

class Depth_net():
    def __init__(self) -> None:
            
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load("depth_net/trained_networks/encoder.pth", map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load("depth_net/trained_networks/depth.pth", map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()


# PREDICTION
    def predict(self, path):
        input_image = PIL.Image.open(path).resize((640, 192)).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((self.feed_width, self.feed_height), PIL.Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(self.device)
        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
        return depth[0,0,:,:].cpu()#, disp_resized
    

