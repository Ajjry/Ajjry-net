function k=GW(im, p)
% Image should be normalized to 0-1

if ~exist('p','var')
    p=6;
end

imP = im.^p;

R_avg = mean2(imP(:,:,1)).^(1/p);
G_avg = mean2(imP(:,:,2)).^(1/p);
B_avg = mean2(imP(:,:,3)).^(1/p);

Avg = mean2(imP).^(1/p);
    
k = [R_avg G_avg B_avg]./Avg;



end

