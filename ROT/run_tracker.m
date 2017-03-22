% The implementation is built upon the code provided by the following
% references
% [1] Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg and Joost van de Weijer.
%     "Adaptive Color Attributes for Real-Time Visual Tracking".
%     Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
% [2] Danelljan, M., H?ger, G., Khan, F., & Felsberg, M. (2014). 
%     "Accurate scale estimation for robust visual tracking". 
%     In British Machine Vision Conference, Nottingham, September 1-5, 2014. BMVA Press.
% [3] Henriques, J. F., Caseiro, R., Martins, P., & Batista, J. (2015). 
%     "High-speed tracking with kernelized correlation filters". 
%     IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(3), 583-596.
% run_tracker.m

close all;clear;
% clear all;
% addpath
addpath(genpath(pwd));
%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'sequences/';
% base_path = 'E:\dxp\papers\(project)videoTracking\(2014data)tracking\vot2014\';

%parameters according to CN [1]
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



params.visualization = 1;
% dsst parameter [2]
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.number_of_scales = 29;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.015;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples
params.scale_learning_rate = 0.025;	%ask the user for the video

%ask the user for the video
video_path = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, ground_truth, video_path] = ...
	load_video_info2013(video_path);

% [img_files, pos, target_sz, ground_truth, video_path] = ...
% 	load_video_infoVot2014(video_path);% for vot2014

params.init_pos = floor(pos) + floor(target_sz/2);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = video_path;

[positions, fps] = color_trackerEM2S2_dhog_kcf3(params);

% calculate precisions
[distance_precision, PASCAL_precision, average_center_location_error] = ...
    compute_performance_measures(positions, ground_truth);

fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
    average_center_location_error, 100*distance_precision, 100*PASCAL_precision, fps);
