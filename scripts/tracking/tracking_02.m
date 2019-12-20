colorImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
binaryImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\PixelLabelData_top");
imds = imageDatastore(binaryImgDir);
% Which image to read. We got a total of 19 images
img_idx = 7;
I = readimage(imds,img_idx);
I = imresize(I,0.1);

% imshow(I, [0 1]) for binary b/w images
% imshow(I, [0 1]);

%%

pathLabels = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\test_csv.csv";
labels = readmatrix(pathLabels)
