imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(imgDir);
% Which image to read. We got a total of 19 images
img_idx = 4
I = readimage(imds,img_idx);
I = imresize(I,0.1);
I = im2double(I);
% [360 24] works well
range = [360 24];

% RGB to HSV conversion
im = rgb2hsv(I);         

% Normalization range between 0 and 1
range = range./360;

% Mask creation
if(size(range,1) > 1), error('Error. Range matriz has too many rows.'); end
if(size(range,2) > 2), error('Error. Range matriz has too many columns.'); end

if(range(1) > range(2))
    % Red hue case
    mask = (im(:,:,1)>range(1) & (im(:,:,1)<=1)) + (im(:,:,1)<range(2) & (im(:,:,1)>=0));
else
    % Regular case
    mask = (im(:,:,1)>range(1)) & (im(:,:,1)<range(2));
end

imshow(I .* mask);