function [imgx] = select2(img)
h = size(img,1);
w = size(img,2);
data11 = mean(img,3);
list = sort(data11(:));
nums=floor(20*h*w/100); 
hhh = data11>=list(end - nums);
imgx = (hhh).*img;

end

