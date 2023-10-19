function [Pillum,Greyidx_unique,mask]= GPqian(img,mask)

img = img./max(img(:));
mask0 =(max(img,[],3)>=0.95) | (sum(img,3)<=0.0315);
mask=mask | mask0;

Npre = 10.^[-1];
Npixels = size(img,1)*size(img,2);
numGPs=floor(Npre*Npixels/100);

delta_threshold=10.^[-4];

img_column=reshape(img,[],3);
r=img(:,:,1); g=img(:,:,2); b=img(:,:,3);

%averagiong
hh = fspecial('average',[7 7]);
r = imfilter(r,hh,'circular');g = imfilter(g,hh,'circular');b = imfilter(b,hh,'circular');

%mask 0 elements
mask=mask | (r==0) | (g==0) | (b==0);
r(r==0)=eps;  g(g==0)=eps;  b(b==0)=eps; norm1=r+g+b;

%mask low contrast pixels
delta_r=DerivGauss(r,.5); delta_g=DerivGauss(g,.5); delta_b=DerivGauss(b,.5);
mask=mask | delta_r<=delta_threshold & delta_g<=delta_threshold & delta_b<=delta_threshold;

ttt = abs(max(max(delta_r,delta_g),delta_b)-min(min(delta_r,delta_g),delta_b));
yy = (delta_r+delta_g)+delta_b;
tt(:,:,1) = delta_r;tt(:,:,2) = delta_g;tt(:,:,3) = delta_b;

log_r=log(r)-log(norm1); log_b=log(b)-log(norm1);
% log_g=log(g)-log(norm1);
% tt(:,:,1) = log_r;tt(:,:,2) = log_g;tt(:,:,3) = log_b;

delta_log_r=DerivGauss(log_r,.5);
delta_log_b=DerivGauss(log_b,.5);
mask=mask | (delta_log_r==Inf) | (delta_log_b==Inf);

data=[(delta_log_r(:)),(delta_log_b(:))];%,Mr(:)-norm1_M(:),Mg(:)-norm1_M(:),Mb(:)-norm1_M(:)];
mink_norm=2;
norm2_data=power(sum(power(data,mink_norm),2),1/mink_norm);
map_uniquelight=reshape(norm2_data,size(delta_log_r));

%(1)mask out some pixels
map_uniquelight(mask==1)=max(map_uniquelight(:));

%(2)average
hh = fspecial('average',[7 7]);
map_uniquelight = imfilter(map_uniquelight,hh,'circular');

%computer map_old_greyness
%map_old_greyness = GetGreyidx(input_im,'GPedge',0.5);

%filter by map_uniquelight
Greyidx_unique=map_uniquelight;

sort_unique=sort(Greyidx_unique(:));
Gidx_unique = zeros(size(Greyidx_unique));

%     %recalculate the numGPs
%     npixels_notmasked=length(find(mask(:)==0));
%     numGPs=floor(Npre*npixels_notmasked/100);

Gidx_unique(Greyidx_unique<=sort_unique(floor(numGPs))) = 1;
choosen_pixels=img_column(Gidx_unique==1,:);
%choosen_pixels=normr(choosen_pixels);
Pillum=normr(mean((choosen_pixels),1));

% tt = 1;
% % img = img./max(img(:));
% R=img(:,:,1); G=img(:,:,2); B=img(:,:,3);
% % R(R==0)=eps; G(G==0)=eps; B(B==0)=eps;
% Lum = R+G+B;
% 
% % Laplacian of Gaussian filter
% loghh = fspecial('log',[5 5]);
% % loghh = [0 -1 0; -1 4 -1; 0 -1 0];
% M1 = imfilter(log(R)-log(Lum),loghh,'circular');
% M2 = imfilter(log(B)-log(Lum),loghh,'circular');
% 
% % M1 = LocalStd(abs(log(R)-log(Lum)),[3 3]);
% % M2 = LocalStd(abs(log(B)-log(Lum)),[3 3]);
% Greyidx = sqrt(M1.^2 + M2.^2);
% 
% % CI = imfilter(Lum,loghh,'circular');
% % CIr = LocalStd(R,[5 5]);
% % CIg = LocalStd(G,[5 5]);
% % CIb = LocalStd(B,[5 5]);
% CIr = abs(imfilter(R,loghh,'circular'));
% CIg = abs(imfilter(G,loghh,'circular'));
% CIb = abs(imfilter(B,loghh,'circular'));
% Greyidx(CIr<0.0001 & CIg<0.0001 & CIb<0.0001)= max(Greyidx(:));
% 
% hh = fspecial('average',[7 7]);
% Greyidx = imfilter(Greyidx,hh,'circular');
% 
% if ~isempty(mask)
%     Greyidx(find(mask)) = max(Greyidx(:));  % dataset: GS568 HDR Greyball
% end
% 
% tt=sort(Greyidx(:));
% Gidx = zeros(size(Greyidx));
% Gidx(Greyidx<=tt(floor(0.1*size(img,1)*size(img,2)/100))) = 1;
% 
% RR = Gidx.*R; GG = Gidx.*G; BB = Gidx.*B;
% Rillu = sum(RR(:)); Gillu = sum(GG(:));Billu = sum(BB(:));
%     
% ee = [Rillu Gillu Billu];
% Pillum = ee./norm(ee,2);

