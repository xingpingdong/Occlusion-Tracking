function [rect_positions, fps] = color_trackerEM2S2_dhog_kcf3_t(params)
% 加入方框放缩处理 引入dsst的方法  使用hog特征，计算分块平均距离 作为判断遮挡阈值 加入KCF
% 将KCF得到的响应值 上采样后 和CN得到的响应值 加权平均 最后求最大值
% [positions, fps] = color_tracker(params)

%% parameters
padding = params.padding;
output_sigma_factor = params.output_sigma_factor;
sigma = params.sigma;
lambda = params.lambda;
learning_rate = params.learning_rate;
compression_learning_rate = params.compression_learning_rate;
non_compressed_features = params.non_compressed_features;
compressed_features = params.compressed_features;
num_compressed_dim = params.num_compressed_dim;

nweights = params.nweights;%[0.5,0.5];
nweights  = reshape(nweights,1,1,[]);
%% 
K = 5;
lambda_l = 10;
s0 = 1; % 初始放缩因子
% scale_d_next = params.scale_d_next;
% scale_Thr = params.scale_Thr;
fre = 1; %更新频率
fre2 = 1; %突变阈值 更新频率
Thr_next = 1; %相邻目标框的距离阈值
sum_Thr1 = 0;
count_Thr1 = 0;


Thr_Occ = 1; %遮挡阈值
Thr_Occ_P = 1; %部分遮挡阈值
Thr_Occ_learning_rate = 0.015;%0.075;
trans_vec = [-1,0;1,0;0,-1;0,1];%4个方向  
trans_vec8 = [-1,0;1,0;0,-1;0,1;-1,-1;1,-1;-1,1;1,1];%8个方向

% video_path = params.video_path;
% img_files = params.img_files;
s_frames = params.s_frames;
res_path = params.res_path;
pos = floor(params.init_pos);
target_sz = floor(params.wsize);

%% addtion
s00 = sqrt(target_sz(1)*target_sz(2)/4000);
s00 = max(s00,1);
% s00=1;
target_sz0 = target_sz;
target_sz = floor(target_sz/s00);
pos = floor(pos/s00);
% window size, taking padding into account
im = imread(s_frames{1});
im = imresize(im,1/s00);
im_sz = size(im);
paddingH = params.paddingH;%struct('generic', 1.5, 'large', 1, 'height', 0.4);
sz = get_search_window(target_sz, im_sz, paddingH);
% sz = floor(target_sz * (1 + padding));

% 用于判断遮挡
cell_size = 8;%floor(min(target_sz)/4);%param.cell_size; %hog特征的范围大小
target_sz_hog = floor(target_sz/cell_size);

%% KCF 参数
kernel.type = 'gaussian';%'polynomial'; 'linear';
	
	features.gray = false;	
kcf.lambda = 1e-4;  %regularization
kcf.output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
kcf.interp_factor = params.kcf.interp_factor;
kcf.padding = padding;%1.5;
kcf_sz = sz;%floor(target_sz * (1 + kcf.padding));
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		features.cell_size = 4;
kcf.output_sigma = sqrt(prod(target_sz)) * kcf.output_sigma_factor / features.cell_size;
	kcf.yf = fft2(gaussian_shaped_labels(kcf.output_sigma, floor(kcf_sz / features.cell_size)));
	%store pre-computed cosine window
	kcf.cos_window = hann(size(kcf.yf,1)) * hann(size(kcf.yf,2))';        
kcf_sz2 = size(kcf.yf);
    %%  dsst 参数
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_max_area = params.scale_model_max_area;
scale_learning_rate = params.scale_learning_rate;
%% dsst 初始化
init_target_sz = target_sz;
base_target_sz = target_sz;
% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
scale_sigma = nScales/sqrt(nScales) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));


% store pre-computed scale filter cosine window
if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));
end;

% scale factors
ss = 1:nScales;
scaleFactors = scale_step.^(ceil(nScales/2) - ss);

% compute the resize dimensions used for feature extraction in the scale
% estimation
scale_model_factor = 1;
if prod(init_target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
end
scale_model_sz = floor(init_target_sz * scale_model_factor);

currentScaleFactor = 1;

% find maximum and minimum scales
[hs,ws,~] = size(im);
min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(scale_step));

%% CN 参数
visualization = params.visualization;

num_frames = numel(s_frames);

% load the normalized Color Name matrix
temp = load('w2crs');
w2c = temp.w2crs;

use_dimensionality_reduction = ~isempty(compressed_features);

% desired output (gaussian shaped), bandwidth proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
yf = single(fft2(y));

% store pre-computed cosine window
cos_window = single(hann(sz(1)) * hann(sz(2))');

% to calculate precision
positions = zeros(numel(s_frames), 4);
rect_positions = zeros(numel(s_frames), 4);

% initialize the projection matrix
projection_matrix = [];

% to calculate fps
time = 0;

% 为建立分类器池（pool）申请空间
% response_pool = zeros(sz(1),sz(2),K);
response_pool = cell(K,1);
% target_pool = zeros(target_sz(1)*target_sz(2)*3,K);
weight_pool = zeros(1,K); 
loss_pool = weight_pool;

% for test
m_response =  zeros(num_frames,1);

for frame = 1:num_frames,
    % load image
    im = imread(s_frames{frame}); 
    im = imresize(im,1/s00);   
    
    sz_s = floor(sz*s0);
    kcf_sz_s = floor(kcf_sz*s0);
    target_sz_s = floor(target_sz*s0);
    tic;
    if frame == 1   
        %%  CN model training 
        % extract the feature map of the local image patch to train the classifer
        [xo_npca, xo_pca] = get_subwindow_s(im, pos, sz_s, sz, non_compressed_features, compressed_features, w2c);
        
        % initialize the appearance
        z_npca = xo_npca;
        z_pca = xo_pca;
        
        % set number of compressed dimensions to maximum if too many
        num_compressed_dim = min(num_compressed_dim, size(xo_pca, 2));    
    
        % if dimensionality reduction is used: update the projection matrix
        if use_dimensionality_reduction
            % compute the mean appearance
            data_mean = mean(z_pca, 1);

            % substract the mean from the appearance to get the data matrix
            data_matrix = bsxfun(@minus, z_pca, data_mean);

            % calculate the covariance matrix
            cov_matrix = 1/(prod(sz) - 1) * (data_matrix' * data_matrix);

            % calculate the principal components (pca_basis) and corresponding variances

                [pca_basis, pca_variances, ~] = svd(cov_matrix);        

            % calculate the projection matrix as the first principal
            % components and extract their corresponding variances
            projection_matrix = pca_basis(:, 1:num_compressed_dim);
            projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);

                % initialize the old covariance matrix using the computed
                % projection matrix and variances
                old_cov_matrix = projection_matrix * projection_variances * projection_matrix';        
        end
    
        % project the features of the new appearance example using the new
        % projection matrix
        x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);

        % calculate the new classifier coefficients
        kf = fft2(dense_gauss_kernel(sigma, x));
        new_alphaf_num = yf .* kf;
        new_alphaf_den = kf .* (kf + lambda);
        % first frame, train with a single image
        alphaf_num = new_alphaf_num;
        alphaf_den = new_alphaf_den;
        %% KCF model training
         kcf_alphaf = [];    kcf_xf = [];
        [kcf_alphaf, kcf_xf] = kcf_track(im, pos, kcf_sz_s ,kcf_sz, kcf, ...
            kernel, features, frame, kcf_alphaf, kcf_xf );
        %% DSST model training
        % extract the training sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        % calculate the scale filter update
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);   
            
        sf_den = new_sf_den;
        sf_num = new_sf_num;
        
         %% 初始化分类器池
         [h,w,d] = size(im);
         
        pointer = 0; indp = pointer+1;
        n_class = 1; n_con = 0; n_con2 = 0;
        % target patch pool
        tpatch = get_patch_s_hog(im, pos, target_sz_s, target_sz, target_sz_hog,cell_size);
        [dp,np] = size(tpatch);
        target_pool = zeros(dp,np,K);        
        target_pool(:,:,indp) = tpatch;
        weight_pool(indp) = 1;    
        % CN model pool    
        w_cn = 1;
        alphaf_num_pool = zeros([size(alphaf_num),K]);
        alphaf_den_pool = zeros([size(alphaf_den),K]);        
        z_npca_pool = zeros([size(z_npca),K]);
        z_pca_pool = zeros([size(z_pca),K]);
        projection_matrix_pool = zeros([size(projection_matrix),K]);
        old_cov_matrix_pool = zeros([size(old_cov_matrix),K]);
        
        alphaf_num_pool(:,:,indp) = alphaf_num;
        alphaf_den_pool(:,:,indp) = alphaf_den;
        z_npca_pool(:,:,indp) = z_npca;
        z_pca_pool(:,:,indp) = z_pca;
        projection_matrix_pool(:,:,indp) = projection_matrix;
        old_cov_matrix_pool(:,:,indp) = old_cov_matrix;
        % DSST model pool
        sf_num_pool = zeros([size(sf_num),K]);
        sf_den_pool = zeros([size(sf_den),K]);  
        
        sf_num_pool(:,:,indp) = sf_num;
        sf_den_pool(:,:,indp) = sf_den;
        % KCF model pool
        w_kcf = 1;
        kcf_alphaf_pool = zeros([size(kcf_alphaf),K]); 
        kcf_xf_pool = zeros([size(kcf_xf),K]);
        
        kcf_alphaf_pool(:,:,indp) = kcf_alphaf; 
        kcf_xf_pool(:,:,:,indp) = kcf_xf;
        % 选择突变与遮挡的阈值
        n_trans = size(trans_vec8,1);
        com_patch = zeros(dp,np,n_trans);
        count = 0;
        for j = 1:n_trans
                pos1 = pos + floor(target_sz).*trans_vec8(j,:);
                if pos1(1)-target_sz(1)/2>0 & pos1(2)-target_sz(2)/2>0 & pos1(1)+target_sz(1)/2<h & pos1(2)+target_sz(2)/2<w
                    patch1 = get_patch_s_hog(im, pos1, target_sz_s, target_sz, target_sz_hog,cell_size);
                    count = count + 1;
                    com_patch(:,:,count) = patch1;
                end
        end
%         d_com = sqrt(sqdist(com_patch(:,:,1:count),tpatch))./n_tp;
        [d_com] =  LW_dist(com_patch(:,:,1:count),tpatch);
        Thr_Occ = 0.95*mean(d_com);
    % 计算目标框之间的平均欧式距离
        
%         d_target = sqdist(target_pool,target_pool);
    end
    
    if frame > 1
       
        
        % extract the feature map of the local image patch
        [xo_npca, xo_pca] = get_subwindow_s(im, pos, sz_s, sz, non_compressed_features, compressed_features, w2c);
        % kcf appearance
        patch = get_patch_s(im, pos, kcf_sz_s, kcf_sz);
		zf = fft2(get_features(patch, features, features.cell_size, kcf.cos_window));
        %% 判断是否发生遮挡，或者预测失败
%         n_tp = length(tpatch);
        temp = target_pool(:,:,1:n_class);
        d_pool = LW_dist(temp,tpatch);
%         d_pool = sqrt(sqdist(temp,tpatch))./n_tp;
        min_d_pool = min(d_pool);
        if min_d_pool>Thr_Occ % 完全遮挡,从分类器池选择最佳分类器           
            
            
            for k = 1:n_class   
                %%%% CN model
                 % do the dimensionality reduction and windowing
                x = feature_projection(xo_npca, xo_pca, projection_matrix_pool(:,:,k), cos_window);
                % compute the compressed learnt appearance
                zp = feature_projection(z_npca_pool(:,:,k), z_pca_pool(:,:,k), projection_matrix_pool(:,:,k), cos_window);

                % calculate the response of the classifier
                kf = fft2(dense_gauss_kernel(sigma, x, zp));
                response_cn = real(ifft2(alphaf_num_pool(:,:,k) .* kf ./ alphaf_den_pool(:,:,k)));
%                 max_cn = max(response_cn(:));
%                 max_cn = max(response_cn(:));
                %%%% KCF model
                %calculate response of the classifier at all shifts
                model_xf = kcf_xf_pool(:,:,:,k);
                model_alphaf = kcf_alphaf_pool(:,:,k);
                switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
                end                
                response_kcf = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
%                 max_kcf = max(response_kcf(:));
                res_layer = zeros([sz,2]);
                res_layer(:,:,1) = response_cn;
%                 res_layer(:,:,2) = imResample(ifftshift(response_kcf),sz,'bilinear');
%                 res_layer(:,:,2) = ifftshift(imResample(response_kcf,sz,'bilinear'));
                res_layer(:,:,2) = imResample(circshift(response_kcf, floor(kcf_sz2(1:2) / 2) ),sz,'bilinear');
%                 res_layer = zeros([kcf_sz2,2]);
%                 res_layer(:,:,1) = ifftshift(imResample(response_cn,kcf_sz2,'bilinear'));
% %                 res_layer(:,:,1) = circshift(imResample(response_cn,kcf_sz2,'bilinear'),-floor(kcf_sz2(1:2) / 2)+1);
%                 res_layer(:,:,2) = response_kcf;
                response = sum(bsxfun(@times, res_layer, nweights), 3);
             
                response_pool{k} = response;
                loss_pool(k) = get_loss2(response,lambda_l);
%                 loss_pool(k) = max(response(:));
            end
            [~,idx] = min(loss_pool);
%             [~,idx] = max(loss_pool);
            % 更新位置
            response = response_pool{idx};
            if max(response(:)) > 0.2
            % target location is at the maximum response
            if size(response,1)==sz(1)
                [row, col] = find(response == max(response(:)), 1);
                new_pos = pos - floor(sz/2) + [row, col];
            else
                [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
                if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                    vert_delta = vert_delta - size(zf,1);
                end
                if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                    horiz_delta = horiz_delta - size(zf,2);
                end
                new_pos = pos + features.cell_size * [vert_delta - 1, horiz_delta - 1];
            end
            % update target patch
            tpatch = target_pool(:,:,idx);
            % update CN pool
            projection_matrix = projection_matrix_pool(:,:,idx);
            z_npca = z_npca_pool(:,:,idx);
            z_pca = z_pca_pool(:,:,idx);
            alphaf_num = alphaf_num_pool(:,:,idx);
            alphaf_den = alphaf_den_pool(:,:,idx);
            old_cov_matrix = old_cov_matrix_pool(:,:,idx);
            % update KCF pool
            kcf_xf = kcf_xf_pool(:,:,:,idx);
            kcf_alphaf = kcf_alphaf_pool(:,:,idx);
            % update DSST pool
            sf_num = sf_num_pool(:,:,idx);
            sf_den = sf_den_pool(:,:,idx);
            else
                % CN model predict
                 zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window);                
                x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);               
                kf = fft2(dense_gauss_kernel(sigma, x, zp));
                response_cn = real(ifft2(alphaf_num .* kf ./ alphaf_den));
%                 max_cn = max(response_cn(:));
                % KCF model predict
                switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, kcf_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, kcf_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, kcf_xf);
                end
                response_kcf = real(ifft2(kcf_alphaf .* kzf));  %equation for fast detection
%                 max_kcf = max(response_kcf(:));

                res_layer = zeros([sz,2]);
                res_layer(:,:,1) = response_cn;
%                 res_layer(:,:,2) = imResample(ifftshift(response_kcf),sz,'bilinear');
%                 res_layer(:,:,2) = ifftshift(imResample(response_kcf,sz,'bilinear'));
                res_layer(:,:,2) = imResample(circshift(response_kcf, floor(kcf_sz2(1:2) / 2) ),sz,'bilinear');

%                 res_layer = zeros([kcf_sz2,2]);
%                 res_layer(:,:,1) = ifftshift(imResample(response_cn,kcf_sz2,'bilinear'));
% %                 res_layer(:,:,1) = circshift(imResample(response_cn,kcf_sz2,'bilinear'),-floor(kcf_sz2(1:2) / 2)+1);
%                 res_layer(:,:,2) = response_kcf;
                response = sum(bsxfun(@times, res_layer, nweights), 3);
                
                if size(response,1)==sz(1)               
%                     response = response_cn;
                    % target location is at the maximum response
                    [row, col] = find(response == max(response(:)), 1);
                    new_pos = pos - floor(sz/2) + [row, col];
                else
%                     response = response_kcf;
                    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
                    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                        vert_delta = vert_delta - size(zf,1);
                    end
                    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                        horiz_delta = horiz_delta - size(zf,2);
                    end
                    new_pos = pos + features.cell_size * [vert_delta - 1, horiz_delta - 1];
                end
            end
        else %无完全遮挡，使用之前的分类器
                 % CN model predict
                 zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window);                
                x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);               
                kf = fft2(dense_gauss_kernel(sigma, x, zp));
                response_cn = real(ifft2(alphaf_num .* kf ./ alphaf_den));
%                 max_cn = max(response_cn(:));
%                 tt0 = sum(response_cn(:));
                % KCF model predict
                switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, kcf_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, kcf_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, kcf_xf);
                end
                response_kcf = real(ifft2(kcf_alphaf .* kzf));  %equation for fast detection
%                 figure(2), imshow(response_kcf);
%                 max_kcf = max(response_kcf(:));

                res_layer = zeros([sz,2]);
                res_layer(:,:,1) = response_cn;
%                 res_layer(:,:,2) = imResample(ifftshift(response_kcf),sz,'bilinear');
%                 res_layer(:,:,2) = ifftshift(imResample(response_kcf,sz,'bilinear'));
                res_layer(:,:,2) = imResample(circshift(response_kcf, floor(kcf_sz2(1:2) / 2) ),sz,'bilinear');
                
%                 res_layer = zeros([kcf_sz2,2]);
%                 res_layer(:,:,1) = ifftshift(imResample(response_cn,kcf_sz2,'bilinear'));
% %                 res_layer(:,:,1) = circshift(imResample(response_cn,kcf_sz2,'bilinear'),-floor(kcf_sz2(1:2) / 2)+1);
%                 res_layer(:,:,2) = response_kcf;
                response = sum(bsxfun(@times, res_layer, nweights), 3);
                if size(response,1)==sz(1)               
%                     response = response_cn;
                    % target location is at the maximum response
                    [row, col] = find(response == max(response(:)), 1);
                    if ~isempty(row)
                        new_pos = pos - floor(sz/2) + [row, col];
                    else
                        new_pos = pos;
                    end
                else
%                     response = response_kcf;
                    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
                    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                        vert_delta = vert_delta - size(zf,1);
                    end
                    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                        horiz_delta = horiz_delta - size(zf,2);
                    end
                    new_pos = pos + features.cell_size * [vert_delta - 1, horiz_delta - 1];
                end
            
%         end
             %% 目标框放缩处理
             pos = new_pos;
             pos = checkPos(pos,hs,ws);
        % extract the test sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        
        % calculate the correlation response of the scale filter
        xsf = fft(xs,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
        response = scale_response;
        
        % find the maximum scale response
        recovered_scale = find(scale_response == max(scale_response(:)), 1);
        
        % update the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
              
        s0 = currentScaleFactor;
        sz_s = floor(sz*s0);
        target_sz_s = floor(target_sz*s0);
        
        end
        
        m_response(frame) = max(response(:));
        
        new_pos = checkPos(new_pos,hs,ws);
        %% 用新得到的目标框 判断是否发生突变
        new_tpatch = get_patch_s_hog(im,new_pos,target_sz_s,target_sz,target_sz_hog,cell_size);
%         new_tpatch = new_tpatch(:);       
       
            d_next = LW_dist(new_tpatch,tpatch);
%             d_next = sqrt(sqdist(tpatch,new_tpatch))/length(new_tpatch);
            pos = new_pos;            
            tpatch = new_tpatch;

        %% 训练当前帧的分类器
        %%%%%%%%%%%%%%%%% update CN model  %%%%%%%%%%%%%%%%%
        [xo_npca, xo_pca] = get_subwindow_s(im, pos, sz_s, sz,non_compressed_features, compressed_features, w2c);
        % update the appearance
        z_npca = (1 - learning_rate) * z_npca + learning_rate * xo_npca;
        z_pca = (1 - learning_rate) * z_pca + learning_rate * xo_pca;
         if use_dimensionality_reduction
            % compute the mean appearance
            data_mean = mean(z_pca, 1);
            % substract the mean from the appearance to get the data matrix
            data_matrix = bsxfun(@minus, z_pca, data_mean);
            % calculate the covariance matrix
            cov_matrix = 1/(prod(sz) - 1) * (data_matrix' * data_matrix);        
            [pca_basis, pca_variances, ~] = svd((1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * cov_matrix);       
        
            % calculate the projection matrix as the first principal
            % components and extract their corresponding variances
            projection_matrix = pca_basis(:, 1:num_compressed_dim);
            projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);       
        
            % update the old covariance matrix using the computed
            % projection matrix and variances
            old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * (projection_matrix * projection_variances * projection_matrix');
        
         end
    
        % project the features of the new appearance example using the new
        % projection matrix
        x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);
        % calculate the new classifier coefficients
        kf = fft2(dense_gauss_kernel(sigma, x));
        new_alphaf_num = yf .* kf;
        new_alphaf_den = kf .* (kf + lambda);    
        % subsequent frames, update the model
        alphaf_num = (1 - learning_rate) * alphaf_num + learning_rate * new_alphaf_num;
        alphaf_den = (1 - learning_rate) * alphaf_den + learning_rate * new_alphaf_den;  
        %%%%%%%%%% update KCF model %%%%%%%%%%%%%%
        [kcf_alphaf, kcf_xf] = kcf_track(im, pos, kcf_sz_s ,kcf_sz, kcf, ...
            kernel, features, frame, kcf_alphaf, kcf_xf );
        
        %%%%%%%%%% update DSST model %%%%%%%%%%%%%%
        % extract the training sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

        % calculate the scale filter update
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);
        
        sf_den = (1 - scale_learning_rate) * sf_den + scale_learning_rate * new_sf_den;
        sf_num = (1 - scale_learning_rate) * sf_num + scale_learning_rate * new_sf_num;
    
        %% 更新分类器池
%         n_tp = length(tpatch);
        temp = target_pool(:,:,1:n_class);
        d_pool = LW_dist(temp,tpatch);
        min_d_pool = min(d_pool);
        if frame <10
        if abs(m_response(frame)-m_response(frame-1)) <0.2 & m_response(frame)>0.3 %& d_next<0.6*Thr_Occ
%             sum_Thr1 = sum_Thr1+0.5;
%             count_Thr1 = count_Thr1+1;
            scale_Thr1 = 0.3;
        else
%             sum_Thr1 = sum_Thr1+0.8;
%             count_Thr1 = count_Thr1+1;
            scale_Thr1 = 0.8;
        end
%         scale_Thr1 = sum_Thr1/count_Thr1;
        end
%         if min_d_pool<0.8*Thr_Occ
%         if Thr_Occ > d_next
            Thr_Occ_p = scale_Thr1*Thr_Occ;
%         else
%             Thr_Occ_p = scale_d_next*d_next;
%         end
        
            
        if min_d_pool<Thr_Occ_p %& d_next<2*Thr_next
%         if min_d_pool<min(scale_d_next*d_next,scale_Thr*Thr_Occ) & d_next<2*Thr_next
%             pointer = 0; indp = pointer+1;
%             n_class = 1; n_con = 0;
            n_con = mod(n_con+1,fre);
            if n_con == fre-1%达到更新频率开始更新
                pointer = mod(pointer+1,K);
                indp = pointer+1;
                n_class = min(n_class+1,K);
                % update target patch;
                target_pool(:,:,indp) = tpatch;
                % update CN pool
                alphaf_num_pool(:,:,indp) = alphaf_num;
                alphaf_den_pool(:,:,indp) = alphaf_den;            
                z_npca_pool(:,:,indp) = z_npca;
                z_pca_pool(:,:,indp) = z_pca;
                projection_matrix_pool(:,:,indp) = projection_matrix;
                old_cov_matrix_pool(:,:,indp) = old_cov_matrix;
                % update KCF pool
                kcf_xf_pool(:,:,:,indp) = kcf_xf;
                kcf_alphaf_pool(:,:,indp) = kcf_alphaf;
                % update DSST pool
                sf_num_pool(:,:,indp) = sf_num;
                sf_den_pool(:,:,indp) = sf_den;
            end   
            n_con2 = mod(n_con2+1,fre2);
            if n_con2 == fre2-1 
                % 选择突变与遮挡的阈值
                n_trans = size(trans_vec8,1);
                com_patch = zeros(dp,np,n_trans);
                count = 0;
                for j = 1:n_trans
                        pos1 = pos + floor(target_sz_s*1).*trans_vec8(j,:);
                        pos1 = max(pos1,floor(target_sz_s/2));
                        pos1 = min(pos1,floor([h,w]-target_sz_s/2));
%                         if pos1(1)-target_sz(1)/2>0 & pos1(2)-target_sz(2)/2>0 & pos1(1)+target_sz(1)/2<h & pos1(2)+target_sz(2)/2<w
                            patch1 = get_patch_s_hog(im, pos1, target_sz_s, target_sz, target_sz_hog,cell_size);
%                             patch_hog = double(fhog(single(patch1) / 255, cell_size, 9));
%                             patch_hog(:,:,end)=[];
                            count = count + 1;
                            com_patch(:,:,count) = patch1;
%                         end
                end
                d_com = LW_dist(com_patch(:,:,1:count),tpatch);
                Thr_Occ = (1-Thr_Occ_learning_rate)*Thr_Occ+Thr_Occ_learning_rate*0.95*min(d_com);
%                 Thr_Occ = 0.9*min(d_com);
                % 部分遮挡阈值
%                 count = 0;
%                 for j = 1:n_trans
%                         pos1 = pos + floor(target_sz*0.2).*trans_vec8(j,:);
%                         if pos1(1)-target_sz(1)/2>0 & pos1(2)-target_sz(2)/2>0 & pos1(1)+target_sz(1)/2<h & pos1(2)+target_sz(2)/2<w
%                             patch1 = get_patch(im, pos1, target_sz);
%                             count = count + 1;
%                             com_patch(:,count) = patch1(:);
%                         end
%                 end
%                 d_com = sqrt(sqdist(com_patch(:,1:count),tpatch))./n_tp;
%                 Thr_Occ_P = 1*min(d_com);
            end
           
            
        else
            n_con = fre-1;
            n_con2 = fre2-1;
        end
        % 更新突变阈值
        Thr_next = 2*d_next;
    end   
   
    
    %save position
%     positions(frame,:) = [pos target_sz_s];
    positions(frame,:) = [floor(pos*s00) floor(target_sz_s*s00)];
%     T_Occ(frame,:) = Thr_Occ;
    
    time = time + toc;
    rect_positions(frame,:) = floor([pos([2,1]) - target_sz_s([2,1])/2, target_sz_s([2,1])]*s00);
    
    %visualization
    if visualization == 1
        rect_position = [pos([2,1]) - target_sz_s([2,1])/2, target_sz_s([2,1])];
        if frame == 1,  %first frame, create GUI
%             figure('Number','off', 'Name',['Tracker - ' video_path]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
%             im_handle = imshow(im, 'Border','tight', 'InitialMag',200);
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position)
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        
        drawnow
%         if frame >100
%         pause
%         end

    end
end

fps = num_frames/time;
% save T_Occ T_Occ
% figure,plot(m_response);
% fprintf('mean response: %f\n',mean(m_response));
% delta_r = m_response(2:end)-m_response(1:end-1);
% figure,plot(delta_r);
% save delta_r delta_r