import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.segmentation import quickshift
import os

class AXAI_BB():
    def __init__(self, inputs):
        self.inputs = inputs
        
    def tensor2cuda(self, tensor):
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
    def mapping2(self,Mask, data_org,K=5, kernel_size=8,max_dist=10, ratio=.1):
        data_org = data_org.squeeze().detach().cpu().numpy()
        image = np.transpose(data_org, (1, 2, 0))
        segments_orig = quickshift(image.astype(np.float), kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
        values, counts = np.unique(segments_orig, return_counts=True)
        attack_frequency=[]
        attack_intensity=[]

        for i in range(len(values)):
            segments_orig_loc=segments_orig==values[i]
            tmp = np.logical_and(segments_orig_loc,Mask)
            attack_frequency.append(np.sum(tmp))
            attack_intensity.append(np.sum(tmp)/counts[i])
            
        top_attack = np.sort(attack_intensity)[::-1][:K]
        zero_filter = np.zeros(np.array(attack_intensity).shape, dtype=bool)
        for i in range(len(top_attack)):
            intensity_filter = attack_intensity == top_attack[i]
            zero_filter = zero_filter+intensity_filter

        strongly_attacked_list = values[zero_filter]
        un_slightly_attacked_list = np.delete(values, strongly_attacked_list)
        strongly_attacked_image = copy.deepcopy(image)
        for x in un_slightly_attacked_list:
            strongly_attacked_image[segments_orig == x] = (255,255,255)
        
        return strongly_attacked_image
            
    def threshold(self,diff, percentage):
        dif_1=copy.deepcopy(diff[0]) 
        dif_2=copy.deepcopy(diff[1])
        dif_3=copy.deepcopy(diff[2])
        dif_total_1 = copy.deepcopy(dif_1)
        dif_total_2 = copy.deepcopy(dif_2)
        dif_total_3 = copy.deepcopy(dif_3)
        thres_1_1=np.percentile(dif_1, percentage)
        thres_1_2=np.percentile(dif_1, 100-percentage)
        mask_1_1 = dif_1 < thres_1_1
        mask_1_2 = (dif_1 >= thres_1_1) & (dif_1 < thres_1_2)
        mask_1_3 = dif_1 >= thres_1_2
        dif_total_1[mask_1_1] = 1
        dif_total_1[mask_1_2] = 0
        dif_total_1[mask_1_3] = 1
        
        thres_2_1=np.percentile(dif_2, percentage)
        thres_2_2=np.percentile(dif_2, 100-percentage)
        mask_2_1 = dif_2 < thres_2_1
        mask_2_2 = (dif_2 >= thres_2_1) & (dif_2 < thres_2_2)
        mask_2_3 = dif_2 >= thres_2_2
        dif_total_2[mask_2_1] = 1
        dif_total_2[mask_2_2] = 0
        dif_total_2[mask_2_3] = 1
        
        thres_3_1=np.percentile(dif_3, percentage)
        thres_3_2=np.percentile(dif_3, 100-percentage)
        mask_3_1 = dif_3 < thres_3_1
        mask_3_2 = (dif_3 >= thres_3_1) & (dif_3 < thres_3_2)
        mask_3_3 = dif_3 >= thres_3_2
        dif_total_3[mask_3_1] = 1
        dif_total_3[mask_3_2] = 0
        dif_total_3[mask_3_3] = 1
        dif_total = dif_total_1+dif_total_2+dif_total_3
        return dif_total