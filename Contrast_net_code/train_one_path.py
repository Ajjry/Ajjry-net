import torch.utils
import numpy as np
import torch.nn as nn
import torchvision
import dataloader
import Enhance
import argparse
import utils
import os
from tqdm import tqdm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#
def train(config):

    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TYB_net = model.enhance_net().cuda()
    # TYB_net.apply(weights_init)
    enhance_net = Enhance.enhance_net(config.chanel, config.kernel_size, device).cuda()

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path, config.ground_truth_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    optimizer=torch.optim.Adam(enhance_net.parameters(),lr=1e-4, weight_decay=config.weight_decay)
    # optimizer_DES = torch.optim.Adam(DES_net.parameters(), lr=1e-4, weight_decay=config.weight_decay)
    # # optimizer_decom = torch.optim.Adam(decom_net.parameters(), lr=1e-5, weight_decay=config.weight_decay)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    # StepLR_DES = torch.optim.lr_scheduler.StepLR(optimizer_DES, step_size=50, gamma=0.5)
    # StepLR_TYB= torch.optim.lr_scheduler.StepLR(optimizer_TYB, step_size=50, gamma=0.5)


    enhance_net.train()
    # DES_net.train()
    # decom_net.train()
    #color_net.train()
    iteration = 0
    #enhance
    Pce_loss = utils.VGGPerceptualLoss()

    #decom
    # L_TV = utils.L_TV_mean()
    start_epoch = 0
    l2_loss = torch.nn.MSELoss()

    # l2_loss = torch.nn.MSELoss()
    if config.resume:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        enhance_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        StepLR.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    for epoch in range(config.num_epochs-start_epoch):
        # ite=0
        for img_lowlight, ground_truth in tqdm(train_loader):
            optimizer.zero_grad()

            iteration += 1
            img_lowlight = img_lowlight.cuda()
            ground_truth = ground_truth.cuda()
            if config.chanel ==1:
                img_lowlight = torch.mean(img_lowlight,1,True)
                ground_truth = torch.mean(ground_truth, 1, True)

            #enhance
            enhance_img,n,bias,gama,kernel,out_local,_= enhance_net(img_lowlight)

            #enhance_loss
            L2_loss_enh = l2_loss(enhance_img,enhance_img)
            contrast_loss = utils.contrast_loss(enhance_img,ground_truth)
            kernel_loss = utils.cal_kernel_loss(kernel)
            loss_enh=L2_loss_enh + contrast_loss + kernel_loss

            pce_loss = Pce_loss(enhance_img, ground_truth)
            #loss_total=loss_enh+5*L2_loss+pce_loss+loss_deno
            loss = loss_enh + pce_loss



            #new

            # optimizer_decom.zero_grad()
            loss.backward()


            optimizer.step()

            # optimizer_decom.step()
            psnr_train = utils.batch_PSNR(enhance_img, ground_truth, 1.)
            ssim_train = utils.SSIM(enhance_img, ground_truth)
            if ((iteration + 1) % config.display_iter) == 0:
                print("epoch:", epoch+start_epoch, ",", "Loss at iteration", iteration + 1, ":", loss.item(), "PSNR:", psnr_train,
                      "SSIM:", ssim_train)

        if ((epoch+start_epoch) % config.snapshot_iter) == 0:
            checkpoint = {
                'epoch': epoch + start_epoch,
                'model': enhance_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': StepLR.state_dict(),

            }

            if not os.path.isdir(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)
            if not os.path.isdir(os.path.join(config.snapshots_folder,'checkpoints')):
                os.mkdir(os.path.join(config.snapshots_folder,'checkpoints'))

            torch.save(checkpoint, os.path.join(config.snapshots_folder,'checkpoints/') + "Epoch" + str(epoch + start_epoch) + '.pth')
            # torch.save(checkpoint_DES, os.path.join(config.snapshots_folder, 'checkpoint_DES/') + "Epoch" + str(epoch + start_epoch) + '.pth')
        # StepLR_decom.step()
        if epoch+start_epoch> 20:
            StepLR.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str,
                        default="/home/cc/CODE/Dehazing_net_code/data/train/OTS3/haze")
    parser.add_argument('--ground_truth_path', type=str, default="/home/cc/CODE/Dehazing_net_code/data/train/OTS3/gt")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--chanel', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--snapshots_folder', type=str, default="/home/cc/CODE/Contrast_net_code/snapshots5")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch199.pth")
    parser.add_argument('--checkpoint', type=str,
                        default='/home/cc/CODE/Contrast_net_code/snapshots5/checkpoints/Epoch56.pth')
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)