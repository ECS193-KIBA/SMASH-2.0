function test()
    bw_segmentation_outline = ReadMaskFromMaskFile('RGB Images/Copy_of_5001_L_10x_mask_only_segmented.tif');
    bw_obj = imcomplement(bw_segmentation_outline);
    pix_size = 0.65;

    % bw_obj = imcomplement(ReadMaskFromMaskFile('RGB Images/AAV21_L_EDL_mhc2a_merged_mask.tif'));
    % pix_size = 0.625;

    label = bwlabel(bw_obj,4);
    num_obj = max(max(label));  % number of segments
    props = regionprops('table',label,'Centroid','Area','Eccentricity','Solidity','Extent','Circularity','PixelIdxList');

    data = table(props.Area,props.Eccentricity,props.Solidity,props.Extent,props.Circularity);  % store segment properties in a table
    cat = categorical(1:num_obj);
    cat(:) = 'Nonfiber';   % assign the category 'fiber' to all segments
    cat = cat';  
    data = addvars(data,cat); % add the category to the dataset
    data.Properties.VariableNames{1} = 'Area';
    data.Properties.VariableNames{2} = 'Eccentricity';
    data.Properties.VariableNames{3} = 'Convexity';
    data.Properties.VariableNames{4} = 'Extent';
    data.Properties.VariableNames{5} = 'Circularity';
    data.Properties.VariableNames{6} = 'Category';

    
    data.Area = pix_size^2 *data.Area;

    model = load('MediumTreeModel.mat','MediumTree');
    classifier = model.MediumTree.ClassificationTree;
    [Class, Score] = predict(classifier,data);
    fiberindex = Class == 'Fiber';
    % onfiberindex = find(Class == 'Nonfiber');
    data.Category(fiberindex) = 'Fiber';
    data = addvars(data, Score(:,1), Score(:,2), 'NewVariableNames',{'Fiber Score','Non Fiber Score'});

    % fiberfiltered = ismember(label,fiberindex);
    % nonfiberfiltered = ismember(label,nonfiberindex);
    
    writetable(data,'TestOutput/test.xlsx')

    for i = 1:num_obj
        imwrite(label == i | bw_segmentation_outline, ['TestOutput/Region', int2str(i), '.png']);
    end
end


function results = ReadMaskFromMaskFile(filename)
    mask = imread(filename);
    graymask = rgb2gray(mask);
    results = imbinarize(graymask,0.99);
end

