%%
clc;	% Clear command window.
clear;	% Delete all variables.
close all;

colorImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(colorImgDir);
imgIdx = 10;
rgbImage = readimage(imds,imgIdx);

% rgbImage = imread('./data/peaches/top/RGB/DSC06230.JPG');

rgbImage = histeq(rgbImage);

hsvImage = rgb2hsv(rgbImage);

hImage = hsvImage(:, :, 1);
sImage = hsvImage(:, :, 2);
vImage = hsvImage(:, :, 3);

%redMap =(hImage < 0.1);
%redHueImg = redMap .* hImage;

%hsvImage(:, :, 1) = redHueImg;


for i = 1:4000
   for j= 1:6000
      if (hImage(i,j) > 0.11)
          vImage(i,j) = 0;
          sImage(i,j) = 0; 
      else
          if (vImage(i,j)< 0.3 || sImage(i,j) < 0.3)
              vImage(i,j) = 0;
              sImage(i,j) = 0;
          else
              sImage(i,j) = 1;
          end
          
      end
   end
    
end

hsvImage(:, :, 1) = hImage;
hsvImage(:, :, 2) = sImage;
hsvImage(:, :, 3) = vImage;

erodImage = hsvImage;
SE = strel('rectangle',[6,6]);
erodImage = imerode(erodImage, SE);

n_rgbImage = hsv2rgb(erodImage);

bin_map = (n_rgbImage(:,:,1) > 0 | n_rgbImage(:,:,2) > 0 | n_rgbImage(:,:,3) > 0);

binImage = bin_map .* n_rgbImage;

%% some enhancements
binImage = imfill(binImage,'holes');
binImage = imfill(binImage,'holes');
binImage = imfill(binImage,'holes');

figure;
imshow(bin_map);

%%
%-------detect centroids based on area-------
figure(3), clf;
imshow(rgbImage);
stats = regionprops(bin_map,'Area', 'Centroid', 'Orientation'); % get region properties
stats_m = mean([stats.Area]); % mean of detected area
stats_max = max([stats.Area]); % mean of detected area

% filter region properties based on area size
lower_thres = 2500;
upper_thres = inf;
idx = find((lower_thres < [stats.Area]) & ([stats.Area] < upper_thres));
stats_cell = struct2cell(stats);
stats_filtered = stats_cell(:,idx);
stats_filtered = cell2struct(stats_filtered, {'Area', 'Centroid', 'Orientation'});

% plot found ROIs
numObj = numel(stats_filtered); % number of objects to plot
hold on 
for k = 1 : numObj
    scatter(stats_filtered(k).Centroid(1), stats_filtered(k).Centroid(2),'b*')
end
hold off

% Die Binärumwandlung ist hier noch etwas rudimentär,
% hätten hierbei auch NDI als Alternative.

% Auf die Centrois kannst mit stats_filtered(i). Centroid(1)  und
% stats_filtered(i). Centroid(2) zugreifen.
