function [im_patch] = get_patch(im, pos, sz)

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

end

