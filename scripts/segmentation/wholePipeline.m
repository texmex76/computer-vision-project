clear all; clc; close all;

% --- hyperparameters ---
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


tracker = trackerGNN( ...
    'FilterInitializationFcn',@initcvekf,...
    'Assignment','Munkres',...
    'TrackLogic','Score',...
    'DetectionProbability',0.8,...
    'AssignmentThreshold',15*[1 Inf],...
    'ConfirmationThreshold',10,...
    'DeletionThreshold',-1,...
    'FalseAlarmRate',1e-8);

positionSelector = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0];

hfig = figure;
hfig.Position = [614   50   631   529];
hfig.Visible = 'on';
tpaxes = axes(hfig);
tp = theaterPlot('Parent',tpaxes,'XLimits',[0 6000], 'YLimits',[0 4000]);
trackP = trackPlotter(tp,'DisplayName','Tracks','HistoryDepth',100,'ColorizeHistory','on','ConnectHistory','on');
    
% --- load data ---
% imgDir = fullfile("C:\Users\Sebastian Gapp\Google Drive\1.Semester\Computer Vision\Lab\3.meeting\data\peaches\top\RGB");
imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(imgDir);

hold on;
% actually there's 19 pictures
for imgIdx=2:17
    I = readimage(imds,imgIdx);
%     imgResized = imresize(I,0.01);
    imshow(I,'Parent',tpaxes);

    % -- apply hue colorfilter ---
    mask = our_colorfilter(I, range);

    % --- to choose/mix pre-processing add-ons ---
    i = mask;  % define input

    % original mask
    mask_0 = i;

    % morphological opening (erosion followed by a dilation)
    SE = strel('disk',5);
    mask_1 = imopen(i, SE);

    % flood-fill operation on background pixels
    mask_2 = imfill(i,'holes');


    % ---detect centroids based on area---
    mask = our_cut_mask(mask_1, min_hight, max_hight);

    [x,y] = our_detection(mask, lower_thres_area, upper_thres_area, lower_thres_eccentricity);
    
    points = cat(2,[x y],ones(size([x y], 1),1));
    detections = {};
    for i=1:size(points,1)
            detections{i} = objectDetection(imgIdx,points(i,:)/100);
    end
    
    [tracks,tentativeTracks,allTracks,analysisInformation] = tracker(detections,imgIdx);
    positions = getTrackPositions(tracks,positionSelector);
    for i=1:size(detections,2)
        x = detections{i}.Measurement(1)*100;
        y = detections{i}.Measurement(2)*100;
%         if exist('plotPts','var') == 1
%             delete(plotPts);
%         end
        plotPts = plot(x, y,'r*','LineWidth',2,'MarkerSize',10);
        plotPts.Color = 'r';
    end
    
    [pos,cov] = getTrackPositions(tracks,[1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]);
    [vel,~] = getTrackVelocities(tracks,[0 1 0 0 0 0;0 0 0 1 0 0;0 0 0 0 0 1]);
    labels = arrayfun(@(x)num2str(x.TrackID),tracks,'UniformOutput',false);
    trackP.plotTrack(pos*100,vel*100,cov*100,labels);
    drawnow
end
hold off;

%%

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
    stats = regionprops(binary_map, 'Area', 'Centroid', 'Eccentricity'); 
    
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