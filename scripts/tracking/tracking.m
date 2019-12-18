color_imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
binary_imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\PixelLabelData_top");
imds = imageDatastore(color_imgDir);
% Which image to read. We got a total of 19 images
img_idx = 7;
I = readimage(imds,img_idx);
I = imresize(I,0.1);

% imshow(I, [0 1]) for binary b/w images
imshow(I);

%%

points = detectMinEigenFeatures(I, 'FilterSize', 9);

% imshow(I, [0 1]);
% hold on;
% for i = 1:size(points.Location, 1)
%    x = points.Location(i, 1);
%    y = points.Location(i, 2);
%    plot(x, y, 'r*', 'LineWidth', 2, 'MarkerSize', 2); 
% end
% title('Detected interest points');
% hold off;

tracker = vision.PointTracker('MaxBidirectionalError',1);

initialize(tracker,points.Location,I);

nextFrame = readimage(imds,img_idx + 1);
nextFrame = imresize(nextFrame,0.1);
[points,validity] = tracker(nextFrame);

imshow(nextFrame, [0 1]);
hold on;
for i = 1:size(points, 1)
   x = points(i, 1);
   y = points(i, 2);
   plot(x, y, 'r*', 'LineWidth', 2, 'MarkerSize', 2); 
end
title('Tracked interest points');
hold off;
