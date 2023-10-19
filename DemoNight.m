clc;clear all;
% addpath('./EdgeBased/');
addpath('F:/NightCC/GPconstancy/');

load('F:/camrea/XeumeiWan/test/gt1.mat');  
main_path='F:/camrea/XeumeiWan/test/img/';
coordpath = 'F:/camrea/XeumeiWan/test/mask/';

Nimg=311; 

Perf = []; 
files = dir(fullfile(main_path,'*.png')); 

for i = 1:Nimg
    fprintf(2,'Processing image %d/%d...\n',i,Nimg);
    img_path = sprintf('%s%d%s',main_path ,i,'.png');
    mask_path = sprintf('%s%d%s',coordpath ,i,'.png');
    img = double(imread(img_path));
%     img = imresize(img, 0.2);
    mask = logical(imread(mask_path)); 
%     mask = imresize(mask, 0.2);
% %==========================GP2015==============================% Yang-CVPR2015
%      Npixels = size(img,1)*size(img,2);
%      numGPs=floor(0.1*Npixels/100); 
%      [outimg,EvaLum] = GPconstancy(img,numGPs,mask);     
%      Perf(i) = angerr(EvaLum,gt1(i,:));  
%      outimg = outimg;
%      imwrite(outimg,'F:/1.png')
%==========================GI2019==============================% Qian-CVPR2019
% %      [Pillum,Greyidx,mask_qian] = GPqian(img,mask);
% % 
%      Npixels = size(img,1)*size(img,2);
%      numGPs=floor(0.1*Npixels/100);
%      nums=floor(20*Npixels/100);
%      [outimg,EvaLum] = RobustGPs(img,numGPs,mask,nums);
% 
%       Perf(i) = angerr(EvaLum,gt1(i,:));  
     
%=========================RobustGP=============================%
     Npixels = size(img,1)*size(img,2);
     numGPs=floor(0.1*Npixels/100); 
     nums=floor(20*Npixels/100);
        img_mask = select(img);
     %%%%%%%%%%%%%%%%%%%%%%%%%%  RGP   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     [outimg,EvaLum] = RobustGP(img,numGPs,mask,nums);
      CorrImg = zeros(size(img));

      CorrImg(:,:,1) = img(:,:,1)./EvaLum(1);
        CorrImg(:,:,2) = img(:,:,2)./EvaLum(2);
        CorrImg(:,:,3) = img(:,:,3)./EvaLum(3);

%      Perf(i) = angerr(EvaLum,gt1(i,:));
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%  GW   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       EvaLum_GW = generl(img,0,1,0,mask);
%       Perf_GW(i) = angerr(EvaLum_GW,gt1(i,:));
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%  GE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      EvaLum_GE = generl(img_mask, 1, 5, 2,mask);
%      Perf_GE(i) = angerr(EvaLum_GE,gt1(i,:));
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%  SOG   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      EvaLum_SOG = generl(img_mask, 0, 5, 0,mask);
%      Perf_SOG(i) = angerr(EvaLum_SOG,gt1(i,:));
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%  maxRGB   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      EvaLum_maxRGB = generl(img_mask, 0, -1, 0,mask);
%      Perf_maxRGB(i) = angerr(EvaLum_maxRGB,gt1(i,:));
% %      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%  WP   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      EvaLum_WP = WP(img_mask);
%      Perf_WP(i) = angerr(EvaLum_WP,gt(i,:));
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

[median(Perf) mean(Perf)]


