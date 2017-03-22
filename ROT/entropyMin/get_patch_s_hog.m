function [patch_hog] = get_patch_s_hog(im, pos, sz, sz0,target_sz_hog,cell_size)

%
% Extracts patch from image im at position pos and
% window size sz. 

if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end

xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
im_patch = double(im(ys, xs, :));
im_patch = imresize(im_patch,sz0);
patch_hog = double(fhog(single(im_patch) / 255, cell_size, 9));
patch_hog(:,:,end)=[];
patch_hog = reshape(patch_hog,prod(target_sz_hog),[]);
patch_hog = patch_hog';
patch_hog = bsxfun(@rdivide,patch_hog,sum(patch_hog,1));%normalize
end

