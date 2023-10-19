function k = GE(im,p,sigma)
% Image should be normalized to 0-1
% n    图像阶数         固定1,这里图像导数的来源有待考证
% p    Minkowski范数    默认6
% alpha 高斯滤波尺度    默认2
%

out = im;

if ~exist('p','var')
    p=6;
end
if ~exist('alpha','var')
    sigma=2;
end


k = fspecial('gaussian',floor(sigma*3+0.5),sigma);%创建高斯模板
im_G = imfilter(im,k,'replicate');%高斯滤波
im_edge = gradient(im_G);%求一阶图像
im_edge = abs(im_edge).^p;%闵可夫斯基p范式

r = im_edge(:,:,1);
g = im_edge(:,:,2);
b = im_edge(:,:,3);

Avg = mean(im_edge(:)).^(1/p);%计算出来的光照颜色

R_avg = mean2(r).^(1/p);%各通道
G_avg = mean2(g).^(1/p);
B_avg = mean2(b).^(1/p);

k = [R_avg G_avg B_avg]./Avg;%增益k

end
