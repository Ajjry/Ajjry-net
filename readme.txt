ColorChecker Dataset for Illuminant Estimation 

The original RAW images are converted with Dcraw
to TIF format without demosaicing. The demosaicing 
is processed with a linear interpolation. The Canon 
5D images have the offset (129) subtracted. This black 
level was estimated by finding the minimum pixel value
across the whole dataset, by Shi and Funt. The black level 
of the Canon 1D images is 0.

The masks images 1, 2 and 3 allow masking respectively 
the ColorChecker in every image, 
the saturated pixels (RGB>3300, following the calculation 
methodology described by Shi and Funt)
and the clipped pixels (pixels with at least 1 clipped 
color channel). 


No gamma correction is applied to the images.
