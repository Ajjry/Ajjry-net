function k = GE(im,p,sigma)
% Image should be normalized to 0-1
% n    ͼ�����         �̶�1,����ͼ��������Դ�д���֤
% p    Minkowski����    Ĭ��6
% alpha ��˹�˲��߶�    Ĭ��2
%

out = im;

if ~exist('p','var')
    p=6;
end
if ~exist('alpha','var')
    sigma=2;
end


k = fspecial('gaussian',floor(sigma*3+0.5),sigma);%������˹ģ��
im_G = imfilter(im,k,'replicate');%��˹�˲�
im_edge = gradient(im_G);%��һ��ͼ��
im_edge = abs(im_edge).^p;%�ɿɷ�˹��p��ʽ

r = im_edge(:,:,1);
g = im_edge(:,:,2);
b = im_edge(:,:,3);

Avg = mean(im_edge(:)).^(1/p);%��������Ĺ�����ɫ

R_avg = mean2(r).^(1/p);%��ͨ��
G_avg = mean2(g).^(1/p);
B_avg = mean2(b).^(1/p);

k = [R_avg G_avg B_avg]./Avg;%����k

end
