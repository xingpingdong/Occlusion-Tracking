function [dcom] =  LW_dist(com_patch,patch)
% compute the locally weighted distance between com_patch and patch
% com_patch: d x n x K
% pathc: d x n

[~,~,K] = size(com_patch);
patch = repmat(patch,[1,1,K]);
temp = com_patch-patch;
% temp = bsxfun(@minus,com_patch,patch);
temp = sum(temp.^2,1);
temp = mean(temp,2);
dcom = temp(:);
end