import torch
import numpy as np
import Enhance
import Decom
import Denoise
import utils
import os
import time
import glob
import cv2
import argparse

h = 0
w = 0
def lowlight(image_path,config):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    if config.train_model=="hdr":
        data_lowlight = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
        data_lowlight = np.asarray(data_lowlight)
        Max = np.max(data_lowlight)
        data_lowlight = np.log(data_lowlight + 1.0)
        A = np.log(Max + 1.0)
        data_lowlight = data_lowlight / A
        data_lowlight_shape = data_lowlight.shape
        if data_lowlight_shape[0] % 8 != 0:
            h = data_lowlight_shape[0] - data_lowlight_shape[0] % 8
        else:
            h = data_lowlight_shape[0]
        if data_lowlight_shape[1] % 8 != 0:
            w = data_lowlight_shape[1] - data_lowlight_shape[1] % 8
        else:
            w = data_lowlight_shape[1]
        data_lowlight = data_lowlight[:h, :w, :]
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = torch.unsqueeze(data_lowlight, 0).to(device)
    else:
        data_lowlight = cv2.imread(image_path)
        data_lowlight_shape = data_lowlight.shape
        if data_lowlight_shape[0] % 8 != 0:
            h = data_lowlight_shape[0] - data_lowlight_shape[0] % 8
        else:
            h = data_lowlight_shape[0]
        if data_lowlight_shape[1] % 8 != 0:
            w = data_lowlight_shape[1] - data_lowlight_shape[1] % 8
        else:
            w = data_lowlight_shape[1]
        data_lowlight = data_lowlight[:h, :w, :]
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight=data_lowlight.permute(2,0,1)
        data_lowlight = torch.unsqueeze(data_lowlight, 0).to(device)


    #model_loading
    La_net=Enhance.enhance_net(config.chanel, 11, device).to(device)
    decom_net=Decom.decompose_net().to(device)
    DES_net=Denoise.Denois_net().to(device)

    snapshots_pth = config.snapshots_pth.replace("snapshots","snapshots_" + config.train_model)
    checkpoint_decom = torch.load(snapshots_pth + "/checkpoint_decom" + "/Epoch" + str(config.checkpoint_Epoch) + ".pth")
    checkpoint_La = torch.load(snapshots_pth + "/checkpoint_La" + "/Epoch" + str(config.checkpoint_Epoch) + ".pth")
    checkpoint_DES = torch.load(snapshots_pth + "/checkpoint_DES" + "/Epoch" + str(config.checkpoint_Epoch) + ".pth")
    La_net.load_state_dict(checkpoint_La['model'])
    DES_net.load_state_dict(checkpoint_DES['model'])
    decom_net.load_state_dict(checkpoint_decom['model'])

    img_lowlight_high_frequency, img_lowlight_low_frequency = decom_net(data_lowlight)

    enhance_img_lowlight_low_frequency, n, bias,gama, kernel,out_local, filter= La_net(img_lowlight_low_frequency)
    print(n,gama)
    # denoise
    denoised_img_lowlight_high_frequency = DES_net(img_lowlight_high_frequency)
    enhanced_image=enhance_img_lowlight_low_frequency+denoised_img_lowlight_high_frequency

    result_path=image_path.replace('vv','result')

    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    utils.save_img(enhanced_image,result_path)

if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_path', type=str, default="/home/cc/CODE/TZC_NET_code/data/test_data/vv/")
        # parser.add_argument('--test_path', type=str, default="/home/cc/CODE/Dehazing_net_code/data/test/SOTS/")
        parser.add_argument('--train_model', type=str, default="haze")
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--chanel', type=int, default="3")
        parser.add_argument('--checkpoint_Epoch', type=int, default="49")
        parser.add_argument('--snapshots_pth', type=str, default="/home/cc/CODE/Contrast_net_code/snapshots")

        config = parser.parse_args()

        file_list = os.listdir(config.test_path)

        for file_name in file_list:

            test_list = glob.glob(config.test_path + file_name )

            for image in test_list:

                lowlight(image,config)




