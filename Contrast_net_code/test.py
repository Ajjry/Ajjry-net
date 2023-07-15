import torch
import numpy as np
import torch.nn as nn
import torchvision
import utils
import glob
import cv2
from raw import *
# import new_model
import Enhance

import scipy.io as scio

h = 0
w = 0


def lowlight(image_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_lowlight = cv2.imread(image_path)
    img = (np.asarray(data_lowlight) / 255.0)
    img_shape = img.shape

    wfenge = int(img_shape[0] / 8)
    hfenge = int(img_shape[1] / 8)
    img = img[:wfenge * 8, :hfenge * 8, :]

    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    data_lowlight = torch.unsqueeze(img, 0).to(device)


    #print(img.shape)

    #print(img.shape)
    #rawim=load_ideal14bit(image_path, width, height)

    TYB_net = Enhance.enhance_net(1, 5, device).to(device)


    load_name = '/home/cc/CODE/Contrast_net_code/snapshots5/checkpoints/Epoch99.pth'
    checkpoint = torch.load(load_name)
    TYB_net.load_state_dict(checkpoint['model'])



    start = time.time()

    data_lowlight = torch.mean(data_lowlight, 1, True)

    # enhance
    enhanced_image, n, bias,gama, kernel,out_local,filter = TYB_net(data_lowlight)
    kernel = kernel.cpu().numpy()
    print(kernel)
    print(n,gama,bias)
    # kernel = np.asarray(kernel.cpu())
    end_time = (time.time() - start)


    print('time:', end_time)


    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    #cv2.imwrite(result_path, results)
    result_path=image_path.replace('SOTS','results')
    # local_path=result_path.replace('result','local')

    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    utils.save_img(enhanced_image, result_path)
    # utils.save_img(out_local, local_path)

if __name__ == '__main__':
    with torch.no_grad():
        filePath = '/home/cc/CODE/Dehazing_net_code/data/test/SOTS/'
        # filePath = '/home/cc/CODE/TZC_NET_code/data/test_data/vv/'
        file_list = os.listdir(filePath)

        for file_name in file_list:

            test_list = glob.glob(filePath + file_name)

            with torch.no_grad():
                for image in test_list:
                    # image = image
                    print(image)
                    lowlight(image)




