clear all; clc; close all;

%% --- hyperparameters ---
% colorfilter
range = [360 30];
% detection
lower_thres_area = 550;
upper_thres_area = inf;
lower_thres_eccentricity = 0.90;
min_hight = 1000; % note: counting starts from top!
max_hight = 3800;

% further parameter: mask
% currently mask_1 is working best: change morphological opening factor

    
%% --- load data ---
% imgDir = fullfile("C:\Users\Sebastian Gapp\Google Drive\1.Semester\Computer Vision\Lab\3.meeting\data\peaches\top\RGB");
imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(imgDir);
img_idx = 11; % Which image to read. We got a total of 19 images
I = readimage(imds,img_idx);

%% -- apply hue colorfilter ---
mask = our_colorfilter(I, range);

%% --- to choose/mix pre-processing add-ons ---
i = mask;  % define input

% original mask
figure(1), clf;
mask_0 = i;
imshow(labeloverlay(I,mask_0,'Transparency',0));

% morphological opening (erosion followed by a dilation)
SE = strel('disk',5)
mask_1 = imopen(i, SE);
figure(2), clf;
imshow(labeloverlay(I,mask_1,'Transparency',0));

% flood-fill operation on background pixels
mask_2 = imfill(i,'holes');
figure(3), clf;
imshow(labeloverlay(I,mask_2,'Transparency',0));


%% ---detect centroids based on area---
mask = our_cut_mask(mask_1, min_hight, max_hight);

[x,y] = our_detection(mask, lower_thres_area, upper_thres_area, lower_thres_eccentricity);

% plot found ROIs
figure(), clf;
grid on
imshow(labeloverlay(I,mask,'Transparency',0));
x_image = size(I, 2);
hold on
plot([0,x_image], [min_hight,min_hight], 'r-');  % min hight
plot([0,x_image], [max_hight,max_hight], 'r-');  % max hight
scatter(x, y, 'r*')
hold off


function [x,y] = our_detection(binary_image, lower_thres_area, upper_thres_area, lower_thres_eccentricity)
% -------------------------------------------
% description: retrive sorted region properties based on binary map (mask)
% input: binary image
%        lower_thres_area: area size
%        upper_thres_area: ||
%        lower_thres_eccentricity: eccentricity of area: ecc = 0: circle, ecc=1:  line segment
%        max_hight: filter elements above hight
% output: coordinates of detected regions (x,y)
% -------------------------------------------
    binary_map = cast(binary_image, 'logical');
    %  returns measurements for the set of properties specified by properties
    %  for each 8-connected component (object) in the binary image
    stats = regionprops(binary_map, 'Area', 'Centroid', 'Eccentricity')    
    
    % filter region properties based on area size and Eccentricity (of ellipse)
    idx = find((lower_thres_area < [stats.Area]) & ([stats.Area] < upper_thres_area) ...
          & ([stats.Eccentricity] < lower_thres_eccentricity));
    stats_cell = struct2cell(stats);
    stats_filtered = stats_cell(:, idx);
    stats_filtered = cell2struct(stats_filtered, {'Area', 'Centroid', 'Eccentricity'});
    
    numObj = numel(stats_filtered); % number of objects
    x = zeros(numObj,1);
    y = zeros(numObj,1);
    
    for k = 1 : numObj
       x(k) = stats_filtered(k).Centroid(1);
       y(k) = stats_filtered(k).Centroid(2);
    end
end                            


function mask = our_colorfilter(image, range)
% -------------------------------------------
% description: modify input image to keep specific hue
% input: image, hue range
% output: mask
% -------------------------------------------
    % equalize lightning differences
    I = histeq(image);

    I = im2double(I);

    % RGB to HSV conversion
    hsv = rgb2hsv(I);         

    % Normalization range between 0 and 1
    range = range./360;

    % Mask creation
    if(range(1) > range(2))
    % Red hue case
        mask = (hsv(:,:,1) > range(1) & (hsv(:,:,1) <= 1)) + (hsv(:,:,1) < range(2) & (hsv(:,:,1) >= 0));
    else
    % Regular case
        mask = (hsv(:,:,1) > range(1)) & (hsv(:,:,1) < range(2));
    end
end


function mask = our_cut_mask(mask, min_hight, max_hight)
% -------------------------------------------
% description: set binary image (mask) above/below threhold zero
% input: mask
% output: mask
% -------------------------------------------
   mask(1:min_hight,:) = 0;
   mask(max_hight:end,:) = 0;
end