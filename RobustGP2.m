function [outimg,EvaLum] = RobustGP2(img,numGPs,mask,nums)
window_szie = 3;
img(img==0)=eps;
R=img(:,:,1); G=img(:,:,2); B=img(:,:,3);
R(R==0)=eps;  G(G==0)=eps;  B(B==0)=eps;
[rr,cc,dd] = size(img);

Lum = (R+G+B)/3;
mask0 =(Lum>=0.95*max(Lum(:))) | (Lum<=0.01*max(Lum(:)));
mask=mask | mask0;

R_m = medfilt2(R,[window_szie,window_szie]);
G_m = medfilt2(G,[window_szie,window_szie]);
B_m = medfilt2(B,[window_szie,window_szie]);
Mr = conflct(R_m,R,window_szie,rr,cc);
Mg = conflct(G_m,G,window_szie,rr,cc);
Mb = conflct(B_m,B,window_szie,rr,cc);

data=[Mr(:),Mg(:),Mb(:)];%6080256*3
Ds= std(data,[],2);
Ds = Ds./(mean(data,2)+eps); %6080256*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bright_Pixel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data1 = [R(:),G(:),B(:)];%6080256*3
% data11 = mean(data1,2);
% list = sort(data11);
% hhh = data11>=list(end - numGPs*200);
% oo = (hhh).*data11;
% Ps = (Ds.*hhh)./(oo+eps);   % 除以亮度  ==> 改进为亮度像素选择  %6080256 1
% Greyidx = reshape(Ps, [rr cc]);
% Greyidx(Greyidx<eps) = max(Greyidx(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    RS      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgGray = mean(img,3);
data11 = sort(imgGray(:)); 
Bright_pixel = imgGray>=data11(end - nums);
fun = @(block_struct) sum(block_struct.data(:));
mask_small = blockproc(Bright_pixel,[20 20],fun);
mask_logical = mask_small>=200;
expand_array = ones(20,20);
mask1 = kron(mask_logical,expand_array);
mask1 = mask1(1:rr,1:cc);
Greyidx = (reshape(Ds, [rr cc]).*mask1);
Greyidx(Greyidx<eps) = max(Greyidx(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

 