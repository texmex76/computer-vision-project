clear all;

colorImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\03\data\peaches\top\RGB");
binaryImgDir = fullfile("C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\PixelLabelData_top");

pathLabelManual = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\peachesManual\";
pathLabelAuto = "C:\Users\User\Google Drive\JKU\WS19\Computer Vision\LAB\project_dir\data\peachesOurDetections\";

pathLabelsBase = pathLabelAuto;

detections = {};
for imgIdx=1:19
    detTemp = {};
    pathLabels = pathLabelsBase + num2str(imgIdx, "%02d") + ".csv";
    points = readmatrix(pathLabels);
    points = cat(2,points,ones(size(points, 1),1));
    for i=1:size(points,1)
            detTemp{i} = objectDetection(imgIdx,points(i,:)/100);
    end
    detections{imgIdx} = detTemp;
end

% ConfirmationThreshold [M,N]:
% Receive at least M detections in the last N updates
% DeletionThreshold [P R]:
% A track is deleted if, in the last R updates, it was assigned less than P detections

% tracker = trackerGNN( ...
%     'FilterInitializationFcn',@initcvekf,...
%     'Assignment','Munkres',...
%     'AssignmentThreshold',100*[1 Inf],...
%     'ConfirmationThreshold',[2 3],...
%     'DeletionThreshold',[10 10]);

tracker = trackerGNN( ...
    'FilterInitializationFcn',@initcvekf)

positionSelector = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0];

hfig = figure;
hfig.Position = [614   50   631   529];
hfig.Visible = 'on';
tpaxes = axes(hfig);
tp = theaterPlot('Parent',tpaxes,'XLimits',[-30 60], 'YLimits',[15 50]);
trackP = trackPlotter(tp,'DisplayName','Tracks','HistoryDepth',100,'ColorizeHistory','on','ConnectHistory','on');

hold on;
global posPrev;
for imgIdx = 2:size(detections,2)-2
    [tracks,tentativeTracks,allTracks,analysisInformation] = tracker(detections{imgIdx},imgIdx);
%     tracks = tracker(detections{imgIdx},imgIdx);
    positions = getTrackPositions(tracks,positionSelector);
    for i=1:size(detections{imgIdx},2)
        x = detections{imgIdx}{i}.Measurement(1);
        y = detections{imgIdx}{i}.Measurement(2);
        plot(x, y,'r*','LineWidth',2,'MarkerSize',10);
    end
%     if size(positions,1) ~= 0
%         for i=1:size(positions,1)
%             x = positions(i,1);
%             y = positions(i,2);
%             plot(x, y,'b*','LineWidth',2,'MarkerSize',5);
%         end
%     end
    [pos,cov] = getTrackPositions(tracks,[1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]);
    [vel,~] = getTrackVelocities(tracks,[0 1 0 0 0 0;0 0 0 1 0 0;0 0 0 0 0 1]);
    labels = arrayfun(@(x)num2str(x.TrackID),tracks,'UniformOutput',false);
    trackP.plotTrack(pos,vel,cov,labels);
    drawnow
    pause(1);
end
hold off;