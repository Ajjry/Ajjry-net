function [img_res] = conflct(img_m,img,window_size,rr,cc)
img_m = reshape(img_m,[1,rr*cc]);
img_m = repmat(img_m,window_size.^2,1);
padding = floor(window_size/2);
img = padarray(img,[padding padding]);
img = im2col(img, [window_size window_size], 'sliding');
img_res = mean(abs(log(img+eps)-log(img_m+eps)));
img_res = col2im(img_res,[1 1],[rr cc],'sliding');
end

