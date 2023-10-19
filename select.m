function [imgx] = select(img)
h = size(img,1);
w = size(img,2);
nums=floor(20*h*w/100); 
I_img = mean(img,3);
data11 = sort(I_img(:)); 
Bright_pixel = I_img>=data11(end - nums);

fun = @(block_struct) sum(block_struct.data(:));
mask_small = blockproc(Bright_pixel,[20 20],fun);
mask_logical = mask_small>=200;
expand_array = ones(20,20);
mask = kron(mask_logical,expand_array);
mask = mask(1:h,1:w);
imgx = mask.*img;
end

