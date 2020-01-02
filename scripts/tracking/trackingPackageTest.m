clear all;

numTracks = 20; % Maximum number of tracks
gate = 45;      % Association gate
vol = 1e9;      % Sensor bin volume
beta = 1e-14;   % Rate of new targets in a unit volume
pd = 0.8;       % Probability of detection
far = 1e-6;     % False alarm rate

tracker = trackerGNN( ...
    'MaxNumTracks', numTracks, ...
    'MaxNumSensors', 1, ...
    'AssignmentThreshold',gate, ...
    'TrackLogic', 'Score', ...
    'DetectionProbability', pd, 'FalseAlarmRate', far, ...
    'Volume', vol, 'Beta', beta);

lin = linspace(109, 100, 10)';
one = ones(10, 1);
measurements = cat(2, lin+randn(10,1)*0.5, lin+randn(10,1)*0.5,one);
measurements2 = cat(2,measurements(:,1),measurements(:,2)-5,one);
time = linspace(1, 10, 10);

hfig = figure;
hfig.Position = [614   50   631   529];
hfig.Visible = 'on';
tpaxes = axes(hfig);
tp = theaterPlot('Parent',tpaxes,'XLimits',[100 110], 'YLimits',[95 110]);
trackP = trackPlotter(tp,'DisplayName','Tracks','HistoryDepth',100,'ColorizeHistory','on','ConnectHistory','on');

positionSelector = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0];

hold on;
global posPrev;
for idx = 1:10
    detection = objectDetection(time(idx),measurements(idx,:));
    detection2 = objectDetection(time(idx),measurements2(idx,:));
    if mod(idx,2) == 0
        detArr = {detection,detection2};
    else
        detArr = {detection2,detection};
    end
    tracks = tracker(detArr,time(idx));
    positions = getTrackPositions(tracks,positionSelector);
    sizePos = size(positions);
    sizePosPrev = size(posPrev);
    sizeDet = size(detArr);
    if sizePos(1) ~= 0 && sizePosPrev(1) ~= 0
        for i=1:sizeDet(2)
            x = detArr{i}.Measurement(1);
            y = detArr{i}.Measurement(2);
            plot(x, y,'r*','LineWidth',2,'MarkerSize',5);
        end
        for i=1:sizePos(1)
            X = [posPrev(i,1) positions(i,1)];
            Y = [posPrev(i,2) positions(i,2)];
            line(X,Y,'Color','blue','LineStyle','-');
        end
    end
    if sizePos(1) ~= 0
        posPrev = positions;
    end
    pause(1);
end
hold off;