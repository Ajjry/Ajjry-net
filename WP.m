function k=WP(im)
% Image should be normalized to 0-1 น้าปปฏ

R_max = max(max(im(:,:,1)));
G_max = max(max(im(:,:,2)));
B_max = max(max(im(:,:,3)));
Max = max(im(:));

k = [R_max G_max B_max]./Max;



end
