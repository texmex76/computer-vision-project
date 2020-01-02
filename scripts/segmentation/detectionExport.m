clc;	% Clear command window.
clear;	% Delete all variables.
close all;

colorImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
pathLabelsBase = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\peachesOurDetections\";

imds = imageDatastore(colorImgDir);

for imgIdx = 1:19
    rgbImage = readimage(imds,imgIdx);

    rgbImage = histeq(rgbImage);
    hsvImage = rgb2hsv(rgbImage);

    hImage = hsvImage(:, :, 1);
    sImage = hsvImage(:, :, 2);
    vImage = hsvImage(:, :, 3);

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

    binImage = imfill(binImage,'holes');
    binImage = imfill(binImage,'holes');
    binImage = imfill(binImage,'holes');

    %-------detect centroids based on area-------
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

    detections = [];
    for i = 1:size(stats_filtered, 1)
        detections(i,1) = int32(stats_filtered(i).Centroid(1));
        detections(i,2) = int32(stats_filtered(i).Centroid(2));
    end

    pathLabels = pathLabelsBase + num2str(imgIdx, "%02d") + ".csv";
    writematrix(detections,pathLabels);
end