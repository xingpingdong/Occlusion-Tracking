function results=run_ROT(seq, res_path, bSaveImage)

% run_tracker.m

close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
% base_path = 'sequences/';

%parameters according to the paper
params.padding = 1.5;         			   % extra area surrounding the target
params.output_sigma_factor = 1/16;		   % spatial bandwidth (proportional to target)
params.sigma = 0.25;         			   % gaussian kernel bandwidth
params.lambda = 1e-2;					   % regularization (denoted "lambda" in the paper)
params.learning_rate = 0.03;%0.052;			   % learning rate for appearance model update scheme (denoted "gamma" in the paper)
params.compression_learning_rate = 0.15;   % learning rate for the adaptive dimensionality reduction (denoted "mu" in the paper)
params.non_compressed_features = {'gray'}; % features that are not compressed, a cell with strings (possible choices: 'gray', 'cn')
params.compressed_features = {'cn'};       % features that are compressed, a cell with strings (possible choices: 'gray', 'cn')
params.num_compressed_dim = 2;             % the dimensionality of the compressed features

% parameters for fusing cn and hog
params.nweights = [0.9,0.1];
params.paddingH = struct('generic', 1.5, 'large', 1, 'height', 0.4);
params.kcf.interp_factor = 0.02;

params = reset_params(seq.name,params);

params.visualization = 0;
% dsst ²ÎÊý
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.number_of_scales = 29;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.015;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples
params.scale_learning_rate = 0.025;	%ask the user for the video

% video_path = choose_video(base_path);
% if isempty(video_path), return, end  %user cancelled
% [img_files, pos, target_sz, ground_truth, video_path] = ...
% 	load_video_info(video_path);

target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);

params.init_pos = floor(pos);
params.wsize = floor(target_sz);
% params.img_files = img_files;
% params.video_path = video_path;
params.s_frames = seq.s_frames;
params.res_path = res_path;
% params.visualization = 0;%bSaveImage;

% video_path = params.video_path;
% img_files = params.img_files;
s_frames = params.s_frames;
im = imread(s_frames{1});
[~,~,ddd] = size(im);
[rect_position, fps] = color_trackerEM2S2_dhog_kcf3_t(params);
% if ddd == 3
%     [rect_position, fps] = color_trackerEM2S2_dhog_t(params);
% else
%     [rect_position, fps] = kcf_trackerEM_t(params);
% end

% calculate precisions
% [distance_precision, PASCAL_precision, average_center_location_error] = ...
%     compute_performance_measures(positions, ground_truth);
% 
% fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
%     average_center_location_error, 100*distance_precision, 100*PASCAL_precision, fps);
disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = rect_position;%each row is a rectangle
results.fps = fps;

end