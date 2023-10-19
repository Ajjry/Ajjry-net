function [outimg,EvaLum] = RobustGPs(img,numGPs,mask,nums) 

img(img==0)=eps;
R=img(:,:,1); G=img(:,:,2); B=img(:,:,3);
R(R==0)=eps;  G(G==0)=eps;  B(B==0)=eps;
[rr,cc,dd] = size(img);

Lum = (R+G+B)/3;
ws = 5;
for i=1:ws
    for j=1:ws
        Larr(:,:,ws*(i-1)+j) = Lum(i:end-(ws-i),j:end-(ws-j),:);
        Rarr(:,:,ws*(i-1)+j) = R(i:end-(ws-i),j:end-(ws-j),:);
        Garr(:,:,ws*(i-1)+j) = G(i:end-(ws-i),j:end-(ws-j),:);
        Barr(:,:,ws*(i-1)+j) = B(i:end-(ws-i),j:end-(ws-j),:);
    end
end

Mlum= median(Larr,3);
idx = Larr>repmat(Mlum,1,1,ws*ws);

Rhigh = zeros(rr,cc);  Rlow = zeros(rr,cc);
Ghigh = zeros(rr,cc);  Glow = zeros(rr,cc);
Bhigh = zeros(rr,cc);  Blow = zeros(rr,cc);

Rhigh(3:end-2,3:end-2) = sum(Rarr.*idx,3)./sum(idx,3);
Rlow(3:end-2,3:end-2) = sum(Rarr.*(1-idx),3)./sum((1-idx),3);

Ghigh(3:end-2,3:end-2) = sum(Garr.*idx,3)./sum(idx,3);
Glow(3:end-2,3:end-2) = sum(Garr.*(1-idx),3)./sum((1-idx),3);

Bhigh(3:end-2,3:end-2) = sum(Barr.*idx,3)./sum(idx,3);
Blow(3:end-2,3:end-2) = sum(Barr.*(1-idx),3)./sum((1-idx),3);
a = Rhigh - Rlow;
%========================Robust + GI============================%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imgGray = mean(img,3);
% data11 = sort(imgGray(:)); 
% Bright_pixel = imgGray>=data11(end - nums);
% fun = @(block_struct) sum(block_struct.data(:));
% mask_small = blockproc(Bright_pixel,[20 20],fun);
% mask_logical = mask_small>=200;
% expand_array = ones(20,20);
% mask1 = kron(mask_logical,expand_array);
% mask1 = mask1(1:rr,1:cc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
log_rhigh=log(Rhigh)-log(Rhigh+Ghigh+Bhigh); log_rlow=log(Rlow)-log(Rlow+Glow+Blow);
log_bhigh=log(Bhigh)-log(Rhigh+Ghigh+Bhigh); log_blow=log(Blow)-log(Rlow+Glow+Blow);

delta_log_r=log_rhigh-log_rlow;
delta_log_b=log_bhigh-log_blow;
mask=mask | (delta_log_r==Inf) | (delta_log_b==Inf);
data=[(delta_log_r(:)),(delta_log_b(:))];%,Mr(:)-norm1_M(:),Mg(:)-norm1_M(:),Mb(:)-norm1_M(:)];
mink_norm=2;
norm2_data=power(sum(power(data,mink_norm),2),1/mink_norm);
Greyidx=reshape(norm2_data,size(delta_log_r));
% Greyidx = (Greyidx.*mask1);
Greyidx(Greyidx<eps) = max(Greyidx(:));
%========================Robust + GP=========================%
% Mr = abs(log(Rhigh)-log(Rlow));
% Mg = abs(log(Ghigh)-log(Glow));
% Mb = abs(log(Bhigh)-log(Blow));
% data=[Mr(:),Mg(:),Mb(:)];
% Ds= std(data,[],2);
% Ds = Ds./(mean(data,2)+eps);
% 
% data1 = [R(:),G(:),B(:)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data11 = sort(mean(data1,2));
% oo = (mean(data1,2)>=data11(end - numGPs*20)).*mean(data1,2);
% Ps = Ds./(oo+eps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ps = Ds./(mean(data1,2)+eps);   % 除以亮度  ==> 改进为亮度像素选择
% Greyidx = reshape(Ps, [rr cc]);
% 
% Greyidx = Greyidx./(max(Greyidx(:))+eps);
% Greyidx(Mr<eps & Mg<eps & Mb<eps)= max(Greyidx(:));
%==========================================================%

if ~isempty(mask)
    Greyidx(find(mask)) = max(Greyidx(:));
end

Greyidx(:,1:5) = max(Greyidx(:));
Greyidx(:,end-4:end) = max(Greyidx(:));
Greyidx(1:5,:) = max(Greyidx(:));
Greyidx(end-4:end,:) = max(Greyidx(:));
Greyidx(isnan(Greyidx)) = max(Greyidx(:));

tt=sort(Greyidx(:));
Gidx = zeros(size(Greyidx));
Gidx(Greyidx<=tt(numGPs)) = 1;

RR = Gidx.*R;
GG = Gidx.*G;
BB = Gidx.*B;

Rillu = sum(RR(:));
Gillu = sum(GG(:));
Billu = sum(BB(:));

EvaLum =[Rillu Gillu Billu];
outimg = Greyidx;
