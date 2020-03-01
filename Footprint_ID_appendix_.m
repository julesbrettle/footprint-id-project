% Identifying Footprints Using Principle Component Analysis
% J. Brettle and T. Jewell-Alibhai
% 2020-03-02

clear all;

% function dir2matrix takes an input directory and locates all
% images with a .jpg extension. It then takes the images, vectorizes
% them and then grayscales them for use with PCA, adding them to a 
% matrix.
[train, train_validate] = dir2matrix('footprint_images');
[test, test_validate] = dir2matrix('footprint_images\Test');

[NN_S, NN_ID] = nearestNeighbors(12,train,test,train_validate,test_validate);

% rowMode() converts vertical cell array to horizontal
NN_S = rowMode(NN_S); 
NN_ID = rowMode(NN_ID);
NN_R = [NN_S'; NN_ID'];

result = vertcat(test_validate,NN_R)


function [mydata, myvalidate] = dir2matrix(Dir)
    % function dir2matrix takes an input directory and locates all
    % images with a .jpg extension. It then takes the images, vectorizes
    % them and then grayscales them for use with PCA, adding them to a 
    % matrix.
    jpegFiles = dir(fullfile(Dir,'*.jpg')); 
    numfiles = length(jpegFiles);
    width = size(imread(fullfile(Dir,jpegFiles(1).name)),1);
    height = size(imread(fullfile(Dir,jpegFiles(1).name)),2);
    mydata = zeros(width*height, numfiles);

    for k = 1:numfiles 
        F = fullfile(Dir,jpegFiles(k).name);
        temp1 = rgb2gray(double(imread(F))./255);
        %imagesc(temp1);
        temp2 = imresize(temp1,[width*height,1]); 
        mydata(:,k) = temp2;
    end
    
    myvalidate = cell(2,numfiles);
    
    for z = 1:numfiles 
        if(length(strfind(jpegFiles(z).name,'C_')) > 0)
            myvalidate(1,z) = {'Cheetah'};
        elseif(length(strfind(jpegFiles(z).name,'BT_')) > 0)
            myvalidate(1,z) = {'Bengal Tiger'};
        elseif(length(strfind(jpegFiles(z).name,'WR_')) > 0)
            myvalidate(1,z) = {'White Rhino'};
        elseif(length(strfind(jpegFiles(z).name,'P_')) > 0)
            myvalidate(1,z) = {'Puma'};
        elseif(length(strfind(jpegFiles(z).name,'L_')) > 0)
            myvalidate(1,z) = {'Leopard'};
        end
        
        temp1 = string(extractBetween(jpegFiles(z).name,"_"," ("));
        myvalidate(2,z) = {temp1(1)};
    end
end

function [NN_S, NN_ID] = nearestNeighbors(numPrincipalComponents, train, test, trainV, testV)

    ATrain = train';
    ATest = test';

    % mean center the train and test data
    ATrain = ATrain - mean(ATrain);
    ATest = ATest - mean(ATest);
    
    % get covariance matrix of the training data
    covar = ATrain'*ATrain;

    % if you wan to just get the eigenvectors with largest eigenvalues
    % you can use eigs. Delta (diagonalized eigenvalues) is not used here.
    [Q, Delta] = eigs(covar, numPrincipalComponents);

    % project into face space
    faceSpaceTrain = Q'*ATrain';
    faceSpaceTest = Q'*ATest';

    % Use nearest neighbor search (you can code this manually if you'd like)
    NN = knnsearch(faceSpaceTrain', faceSpaceTest', 'K',1);
    NN_S = cell(size(NN));
    NN_ID = cell(size(NN));
    for x = 1:size(NN,1)
        NN_S(x,1:size(NN,2)) = trainV(1,NN(x,:));
    end
    for y = 1:size(NN,1)
        NN_ID(y,1:size(NN,2)) = cellstr(string(trainV(2,NN(y,:))));
    end
end

function a = rowMode(matrix)
    % rowMode() converts vertical cell array to horizontal
    a = cell([size(matrix,1) 1]);
    for x = 1:size(matrix,1)
        y = unique(matrix(x,:));
        n = zeros(length(y), 1);
        for iy = 1:length(y)
            n(iy) = length(find(strcmp(y{iy}, matrix(x,:))));
        end
        [~, itemp] = max(n);
        a(x,:) = y(itemp);
    end
    a;
end
