function [outimg,EvaLum] = RobustGP(img,numGPs,mask,nums) 

img(img==0)=eps;
R=img(:,:,1); G=img(:,:,2); B=img(:,:,3);
R(R==0)=eps;  G(G==0)=eps;  B(B==0)=eps;
[rr,cc,dd] = size(img);

Lum = (R+G+B)/3;
mask0 =(Lum>=0.95*max(Lum(:))) | (Lum<=0.01*max(Lum(:)));
mask=mask | mask0;
ws = 5;
for i=1:ws
    for j=1:ws
        Larr(:,:,ws*(i-1)+j) = Lum(i:end-(ws-i),j:end-(ws-j),:) ;% 2012*3012*25
        Rarr(:,:,ws*(i-1)+j) = R(i:end-(ws-i),j:end-(ws-j),:);
        Garr(:,:,ws*(i-1)+j) = G(i:end-(ws-i),j:end-(ws-j),:);
        Barr(:,:,ws*(i-1)+j) = B(i:end-(ws-i),j:end-(ws-j),:);
    end
end
 
Mlum = median(Larr,3);%2012*3012
 
idx = Larr>repmat(Mlum,1,1,ws*ws);%2012*3012*25

Rhigh = zeros(rr,cc);  Rlow = zeros(rr,cc);
Ghigh = zeros(rr,cc);  Glow = zeros(rr,cc);
Bhigh = zeros(rr,cc);  Blow = zeros(rr,cc);

Rhigh(3:end-2,3:end-2) = sum(Rarr.*idx,3)./sum(idx,3);
Rlow(3:end-2,3:end-2) = sum(Rarr.*(1-idx),3)./sum((1-idx),3);

Ghigh(3:end-2,3:end-2) = sum(Garr.*idx,3)./sum(idx,3);
Glow(3:end-2,3:end-2) = sum(Garr.*(1-idx),3)./sum((1-idx),3);

Bhigh(3:end-2,3:end-2) = sum(Barr.*idx,3)./sum(idx,3);
Blow(3:end-2,3:end-2) = sum(Barr.*(1-idx),3)./sum((1-idx),3);

%========================Robust + GI============================%
% log_rhigh=log(Rhigh)-log(Rhigh+Ghigh+Bhigh); log_rlow=log(Rlow)-log(Rlow+Glow+Blow);
% log_bhigh=log(Bhigh)-log(Rhigh+Ghigh+Bhigh); log_blow=log(Blow)-log(Rlow+Glow+Blow);
% 
% delta_log_r=log_rhigh-log_rlow;
% delta_log_b=log_bhigh-log_blow;
% mask=mask | (delta_log_r==Inf) | (delta_log_b==Inf);
% data=[(delta_log_r(:)),(delta_log_b(:))];%,Mr(:)-norm1_M(:),Mg(:)-norm1_M(:),Mb(:)-norm1_M(:)];
% mink_norm=2;
% norm2_data=power(sum(power(data,mink_norm),2),1/mink_norm);
% Greyidx=reshape(norm2_data,size(delta_log_r));

%========================Robust + GP=========================%
Mr = abs(log(Rhigh + eps)-log(Rlow + eps));%2016*3016
Mg = abs(log(Ghigh+ eps)-log(Glow+ eps));
Mb = abs(log(Bhigh+ eps)-log(Blow+ eps));
data=[Mr(:),Mg(:),Mb(:)];%6080256*3
Ds= std(data,[],2);
Ds = Ds./(mean(data,2)+eps); %6080256*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bright_Pixel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data1 = [R(:),G(:),B(:)];%6080256*3
data11 = mean(data1,2);
list = sort(data11);
hhh = data11>=list(end - numGPs*200);
oo = (hhh).*data11;
Ps = (Ds.*hhh)./(oo+eps);   % 除以亮度  ==> 改进为亮度像素选择  %6080256 1
Greyidx = reshape(Ps, [rr cc]);
Greyidx(Greyidx<eps) = max(Greyidx(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    RS      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imgGray = mean(img,3);
% data11 = sort(imgGray(:)); 
% Bright_pixel = imgGray>=data11(end - nums);
% fun = @(block_struct) sum(block_struct.data(:));
% mask_small = blockproc(Bright_pixel,[20 20],fun);
% mask_logical = mask_small>=200;
% expand_array = ones(20,20);
% mask1 = kron(mask_logical,expand_array);
% mask1 = mask1(1:rr,1:cc);
% Greyidx = (reshape(Ds, [rr cc]).*mask1);
% Greyidx(Greyidx<eps) = max(Greyidx(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% data1 = [R(:),G(:),B(:)];
% Ps = Ds./(mean(data1,2)+eps);   % 除以亮度  ==> 改进为亮度像素选择
% Greyidx = reshape(Ps, [rr cc]);
Greyidx = Greyidx./(max(Greyidx(:))+eps);%归1
Greyidx(Mr<eps & Mg<eps & Mb<eps)= max(Greyidx(:));%均值为0的像素赋值为最大GI
%==========================================================%

if ~isempty(mask)
    Greyidx(find(mask)) = max(Greyidx(:));%colorChecker赋值为最大GI
end

Greyidx(:,1:5) = max(Greyidx(:));%前5列赋值最大GI
Greyidx(:,end-4:end) = max(Greyidx(:));%后5列赋值最大GI
Greyidx(1:5,:) = max(Greyidx(:));%前5行赋值最大GI
Greyidx(end-4:end,:) = max(Greyidx(:));%后5行赋值最大GI
Greyidx(isnan(Greyidx)) = max(Greyidx(:));%Nan赋值为最大GI

tt=sort(Greyidx(:));
EvaLum = zeros(length(numGPs),3);

for kk = 1:length(numGPs)
    Gidx = zeros(size(Greyidx));
    Gidx(Greyidx<=tt(numGPs(kk))) = 1;
    RR = Gidx.*R;
    GG = Gidx.*G;
    BB = Gidx.*B;
    
    Rillu = sum(RR(:));
    Gillu = sum(GG(:));
    Billu = sum(BB(:));

    EvaLum(kk,:) =[Rillu Gillu Billu];
end

outimg = Greyidx;
