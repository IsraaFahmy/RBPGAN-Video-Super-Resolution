# tecogan_nets imports
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from utils.net_utils import initialize_weights
from utils.data_utils import float32_to_uint8
from metrics.model_summary import register, parse_model_info
from utils.base_utils import log_info
from .rbpn import Net as RBPN
from .tecogan_nets import FNet

# TODO: Make sure of logic, especially in calculating flow and neighbors

class RBPN_RBPGAN(BaseSequenceGenerator):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, degradation, scale_factor):
        super(RBPN_RBPGAN, self).__init__()

        self.scale = scale_factor
        self.upsample_func = get_upsampling_func(self.scale, degradation)
        
        #FNet: For Optical Estimation
        self.fnet = FNet(num_channels)
        self.RBPN = RBPN(num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor)
    
    def forward(self, lr_data, device=None):
        if self.training:
            out = self.forward_sequence(lr_data)
        else:
            out = self.infer_sequence(lr_data, device)

        return out

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """
        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)  #from zero to before last
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w) #from 1 to last
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w) #return it to 5 dimensions

        # compute the hr data
        hr_data = []
        for i in range(t):
            # prepare current frame, its neighbor and flow
            lr_i = lr_data[:, i, ...]
            neighbor = [lr_data[:, j, ...] for j in range(t) if j != i]
            flow = [self.fnet(lr_i, neighbor_temp) for neighbor_temp in neighbor]

            # compute hr_curr
            hr_curr = self.RBPN(lr_i, neighbor, flow)

            # save and update
            hr_data.append(hr_curr)

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w     

        ret_dict = {
        'hr_data': hr_data,  # n,t,c,hr_h,hr_w
        'hr_flow': hr_flow,  # n,t-1,2,hr_h,hr_w
        'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
        'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
        'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def choose_neighbors(self, ref_frm_index, tot_frm, nFrames):
        tt = int(nFrames/2)

        start = ref_frm_index - tt
        if nFrames%2 == 0:
            end = ref_frm_index + tt
        else:
            end = ref_frm_index + tt + 1
        
        if start < 0:
            end = end - start
            start = 0
        elif end > tot_frm:
            start = start - end + tot_frm
            end = tot_frm

        seq = [x for x in range(start, end) if x!=ref_frm_index]
        return seq

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # set params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale
        nFrames = self.RBPN.nFrames

        # forward
        hr_seq = []

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i: i + 1, ...].to(device)
                neighbor_indices = self.choose_neighbors(i, tot_frm, nFrames)
                neighbor = [lr_data[j: j + 1, ...].to(device) for j in neighbor_indices]
                flow = [self.fnet(lr_curr, neighbor_temp).to(device) for neighbor_temp in neighbor]

                # pad if size is not a multiple of 8
                pad_h = lr_curr.size(2) - lr_curr.size(2)//8*8
                pad_w = lr_curr.size(3) - lr_curr.size(3)//8*8
                flow_pad = [F.pad(flow[j], (0, pad_w, 0, pad_h), 'reflect') for j in range(len(flow))]
                
                # generate hr frame
                hr_curr = self.RBPN(lr_curr, neighbor, flow_pad)

                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc

    # TODO: Adapt for profile: generate_dummy_data and profile