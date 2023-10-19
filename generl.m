function [EvaLum] = generl(input_data,njet,mink_norm,sigma,mask_im)

if(nargin<2), njet=0; end
if(nargin<3), mink_norm=1; end
if(nargin<4), sigma=1; end
if(nargin<5), mask_im=zeros(size(input_data,1),size(input_data,2)); end
% 
% remove all saturated points
Lum = mean(input_data,3);
mask0 =(Lum>=0.95*max(Lum(:))) | (Lum<=0.01*max(Lum(:)));
mask_im=mask_im | mask0;

saturation_threshold = 999999;
mask_im2 = mask_im + (dilation33(double(max(input_data,[],3)>=saturation_threshold)));   
mask_im2=double(mask_im2==0);
mask_im2=set_border(mask_im2,sigma+1,0);
% the mask_im2 contains pixels higher saturation_threshold and which are
% not included in mask_im.


if(njet==0)
   if(sigma~=0)
     for ii=1:3
        input_data(:,:,ii)=gDer(input_data(:,:,ii),sigma,0,0);
     end
   end
end

if(njet>0)
    [Rx,Gx,Bx]=norm_derivative(input_data, sigma, njet);
    
    input_data(:,:,1)=Rx;
    input_data(:,:,2)=Gx;
    input_data(:,:,3)=Bx;    
end

input_data=abs(input_data);

if(mink_norm~=-1)          % minkowski norm = (1,infinity >
    kleur=power(input_data,mink_norm);
%  
    white_R = power(sum(sum(kleur(:,:,1).*mask_im2)),1/mink_norm);
    white_G = power(sum(sum(kleur(:,:,2).*mask_im2)),1/mink_norm);
    white_B = power(sum(sum(kleur(:,:,3).*mask_im2)),1/mink_norm);

    som=sqrt(white_R^2+white_G^2+white_B^2);
    if som ~= 0
        white_R=white_R/som;
        white_G=white_G/som;
        white_B=white_B/som;
    end
else                    %minkowski-norm is infinit: Max-algorithm     
    R=input_data(:,:,1);
    G=input_data(:,:,2);
    B=input_data(:,:,3);
    
    white_R=max(R(:).*mask_im2(:));
    white_G=max(G(:).*mask_im2(:));
    white_B=max(B(:).*mask_im2(:));
    
    som=sqrt(white_R^2+white_G^2+white_B^2);

    white_R=white_R/som;
    white_G=white_G/som;
    white_B=white_B/som;
   
end
EvaLum = [white_R ,white_G ,white_B]; 
end