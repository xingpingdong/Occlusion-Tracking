function [out_npca, out_pca,out_hog] = get_subwindow_s1(im, pos, sz, sz0, non_pca_features, pca_features, w2c, hog_param)

% [out_npca, out_pca] = get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c)
%
% Extracts the non-PCA and PCA features from image im at position pos and
% window size sz. The features are given in non_pca_features and
% pca_features. out_npca is the window of non-PCA features and out_pca is
% the PCA-features reshaped to [prod(sz) num_pca_feature_dim]. w2c is the
% Color Names matrix if used.

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
im_patch = im(ys, xs, :);
im_patch = imresize(im_patch,sz0);

% compute hog feature map
if ~isempty(hog_param)
    cell_size = hog_param.cell_size;
    out_hog = double(fhog(single(im_patch) / 255, cell_size, hog_param.hog_orientations));
    out_hog(:,:,end)=[];
    cos_window = hog_param.cos_window;
    out_hog = bsxfun(@times, out_hog, cos_window);
    sz0 = floor(sz0./cell_size);
    im_patch = imresize(im_patch,sz0);
%     out_hog = reshape(out_hog, [prod(sz0), size(out_hog, 3)]);
else
    out_hog = [];
%     cell_size = 1;
end


% compute non-pca feature map
if ~isempty(non_pca_features)
    out_npca = get_feature_map(im_patch, non_pca_features, w2c);
else
    out_npca = [];
end

% compute pca feature map
if ~isempty(pca_features)          
    temp_pca = get_feature_map(im_patch, pca_features, w2c);
    out_pca = reshape(temp_pca, [prod(sz0), size(temp_pca, 3)]);
else
    out_pca = [];
end


end

