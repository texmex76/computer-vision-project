clear all;

pathLabelManual = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\peachesManual\";
pathLabelAuto = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\peachesOurDetections\";

pathLabelsBase = pathLabelManual;

f = figure('position', [400, 200, 600, 400]);
title("All points, blue truth, red detected");
hold on;

for j = 1:19
    pathLabels = pathLabelManual + num2str(j, "%02d") + ".csv";
    plotPoints(pathLabels, 'b*');
    pathLabels = pathLabelAuto + num2str(j, "%02d") + ".csv";
    plotPoints(pathLabels, 'r*');
end
hold off;

function plotPoints(pathLabels, color)
    points = readmatrix(pathLabels);
    if points ~= 0
        for i = 1:size(points, 1)
           x = points(i, 1);
           y = points(i, 2);
           plot(x, y, color, 'LineWidth', 2, 'MarkerSize', 15); 
        end
    end
end