% Script overlay the manually labeled peaches over the images

clear all; clc; close all; % clean up!

%%

% Define image location
imgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
imds = imageDatastore(imgDir);

%%

% show a single image
I = readimage(imds,5);
I = histeq(I);
imshow(I)

%%

classes = ["peach"];

labelIDs = { ...
    % "peach"
    [
    1; ... % "peach"
    ]
    };

labelDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\PixelLabelData_top");
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%%

img_idx = 13
C = readimage(pxds,img_idx);
cmap = peachColorMap;
B = labeloverlay(readimage(imds,img_idx),C,'ColorMap',cmap);
imshow(B)

%%

function cmap = peachColorMap()
% Define the colormap used by peaches_top dataset.

cmap = [
    153 0 153   % peach
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end
