clc;clear all;
% addpath('./EdgeBased/');
addpath('F:/NightCC/GPconstancy/');

load('F:/camrea/XeumeiWan/test/gt.mat');  
main_path='F:/camrea/XeumeiWan/test/img/';
coordpath = 'F:/camrea/XeumeiWan/test/mask/';

Nimg=513; 

Npre=[0.001 0.01 0.1 1 10];
Perf = zeros(Nimg,length(Npre)+1);
files = dir(fullfile(main_path,'*.png')); 

for i = 1:Nimg
    
    fprintf(2,'Processing image %d/%d...\n',i,Nimg);
    img_path = sprintf('%s%d%s',main_path ,i,'.png');
    mask_path = sprintf('%s%d%s',coordpath ,i,'.png');
    img = double(imread(img_path));
%     img = imresize(img, 0.2);
    mask = logical(imread(mask_path)); 
   
%==========================GP2015==============================% Yang-CVPR2015
%      Npixels = size(img,1)*size(img,2);
%      numGPs=floor(0.01*Npixels/100); 
%      [outimg,EvaLum] = GPconstancy(img,numGPs,mask);     
%      Perf(i) = angerr(EvaLum,gt(i,:));  

%==========================GI2019==============================% Qian-CVPR2019
%      [Pillum,Greyidx,mask_qian] = GPqian(img,mask);
%      Perf(i) = angerr(Pillum,gt(i,:));  
     
%=========================RobustGP=============================%
    Npixels = size(img,1)*size(img,2);
    nums=floor(20*Npixels/100);

    numGPs=[1 floor(Npre*Npixels/100)];
    [outimg,EvaLum] = RobustGP2(img,numGPs,mask,nums);  
          CorrImg = zeros(size(img));

        CorrImg(:,:,1) = img(:,:,1)./EvaLum(1);
        CorrImg(:,:,2) = img(:,:,2)./EvaLum(2);
        CorrImg(:,:,3) = img(:,:,3)./EvaLum(3);
    for kk =1:length(numGPs)
        arr = angerr(EvaLum(kk,:),gt(i,:)); %  groundtruth_illuminants  real_rgb
        Perf(i,kk) = arr;
    end
end

fprintf(1,'The mean error=%f\n',mean(Perf(:,2)));
fprintf(1,'The median error=%f\n',median(Perf(:,2)));


