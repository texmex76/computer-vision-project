clear all; clc; close all;

% --- hyperparameters for the detection ---
% colorfilter
range = [360 30];
% detection
lower_thres_area = 550;
upper_thres_area = inf;
lower_thres_eccentricity = 0.90;
min_hight = 1000; % note: counting starts from top!
max_hight = 3800;

% Further parameter: mask
% In this implementation, mask_1 is used
% Currently mask_1 is working best: change morphological opening factor

% --- hyperparameters for the tracking ---
tracker = trackerGNN( ...
    'FilterInitializationFcn',@initcvekf,...
    'Assignment','Munkres',...
    'TrackLogic','Score',...
    'DetectionProbability',0.65,...
    'AssignmentThreshold',30*[1 Inf],...
    'ConfirmationThreshold',10,...
    'DeletionThreshold',-3,...
    'FalseAlarmRate',3e-5);

% Matrix that we need to extract the positions of the tracks
positionSelector = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0];

% Setting up the plot. We're plotting three things: image, detected peaches
% and tracks
hfig = figure;
hfig.Position = [614   50   631   529];
hfig.Visible = 'on';
tpaxes = axes(hfig);
tp = theaterPlot('Parent',tpaxes,'XLimits',[0 6000], 'YLimits',[0 4000]);
trackP = trackPlotter(tp,'DisplayName','Tracks','HistoryDepth',100,'ColorizeHistory','on','ConnectHistory','on');
    
% --- load data ---
% Specify image path here. We use the top row of the peaches
imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(imgDir);

hold on;
% For-loop over the images
% Actually there's 19 pictures, however in the first and in the last two
% there's no peaches on our tree visible so we'll omit those images
for imgIdx=2:17
    I = readimage(imds,imgIdx);
    % Plotting the image
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
    
    % Now that we have the points of our peaches, let's put them into our
    % tracker. The tracker actually wants 3-dimensional points, so we'll
    % just append a row of ones
    points = cat(2,[x y],ones(size([x y], 1),1));
    % The tracker only takes special `objectDetection` objects. The content
    % of those are the actual points and the time. For the time, we use the
    % image index
    % We also have to scale the point coordinates, otherwise the tracker
    % won't work for some reason, so we just divide over 100
    detections = {};
    for i=1:size(points,1)
            detections{i} = objectDetection(imgIdx,points(i,:)/100);
    end
    
    % Passing our detections and the time to the tracker which will update
    [tracks,tentativeTracks,allTracks,analysisInformation] = tracker(detections,imgIdx);
    % As mentioned before, we need a special matrix to extract the tracked
    % positions
    positions = getTrackPositions(tracks,positionSelector);
    % Here we're looping over our detected points and plot them
    for i=1:size(detections,2)
        % Since the points are scaled down here, we have to multiply them
        % with 100 again
        x = detections{i}.Measurement(1)*100;
        y = detections{i}.Measurement(2)*100;
        plotPts = plot(x, y,'r*','LineWidth',2,'MarkerSize',10);
        plotPts.Color = 'r';
    end
    
    % Here we're plotting the tracks
    [pos,cov] = getTrackPositions(tracks,[1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]);
    [vel,~] = getTrackVelocities(tracks,[0 1 0 0 0 0;0 0 0 1 0 0;0 0 0 0 0 1]);
    labels = arrayfun(@(x)num2str(x.TrackID),tracks,'UniformOutput',false);
    % Multiplying by 100 since the points are actually scaled down
    trackP.plotTrack(pos*100,vel*100,cov*100,labels);
    drawnow
end
hold off;
size(size(tracks.TrackID))

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