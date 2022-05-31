classdef SMASH_ML_1_2_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        FiberTypingControlPanel         matlab.ui.container.Panel
        FiberTypingFileWriteStatusLabel  matlab.ui.control.Label
        FiberTypingDescription_2        matlab.ui.control.Label
        FiberTypingDescription          matlab.ui.control.Label
        DoneFiberTyping                 matlab.ui.control.Button
        WritetoExcelFT                  matlab.ui.control.Button
        CalculateFiberTyping            matlab.ui.control.Button
        FiberTypeColorDropDown          matlab.ui.control.DropDown
        FiberTypeColorDropDownLabel     matlab.ui.control.Label
        FiberTypingDataOutputFolder     matlab.ui.control.EditField
        DataOutputFolderEditField_3Label  matlab.ui.control.Label
        PixelSizeFiberTyping            matlab.ui.control.NumericEditField
        PixelSizeumpixelEditField_3Label  matlab.ui.control.Label
        FiberTypingChannelColorBox      matlab.ui.control.UIAxes
        PropertiesPanel                 matlab.ui.container.Panel
        FiberSizeAxes                   matlab.ui.control.UIAxes
        FeretAxes                       matlab.ui.control.UIAxes
        NonfiberClassificationPanel     matlab.ui.container.Panel
        Label_3                         matlab.ui.control.Label
        PositiveNonfiberObjectsLabel    matlab.ui.control.Label
        OriginalImageLabel              matlab.ui.control.Label
        PercentPositiveTextArea         matlab.ui.control.TextArea
        PercentPositiveLabel            matlab.ui.control.Label
        NonfiberClassificationAccept    matlab.ui.control.Button
        NonfiberClassificationAdjust    matlab.ui.control.Button
        NonfiberClassificationThreshold  matlab.ui.control.NumericEditField
        ThresholdEditField_2Label_3     matlab.ui.control.Label
        NonfiberClassificationAxes_L    matlab.ui.control.UIAxes
        NonfiberClassificationAxes_R    matlab.ui.control.UIAxes
        FiberTypingPanel                matlab.ui.container.Panel
        Label_2                         matlab.ui.control.Label
        AcceptButton                    matlab.ui.control.Button
        AdjustButton                    matlab.ui.control.Button
        ThresholdEditField              matlab.ui.control.NumericEditField
        ThresholdEditFieldLabel         matlab.ui.control.Label
        FThistR                         matlab.ui.control.UIAxes
        FThistL                         matlab.ui.control.UIAxes
        FiberTypingAxesR                matlab.ui.control.UIAxes
        FiberTypingAxesL                matlab.ui.control.UIAxes
        CentralNucleiPanel              matlab.ui.container.Panel
        Label                           matlab.ui.control.Label
        AcceptCentralNuclei             matlab.ui.control.Button
        AdjustCentralNuclei             matlab.ui.control.Button
        ThresholdCentralNuclei          matlab.ui.control.NumericEditField
        ThresholdEditField_2Label       matlab.ui.control.Label
        CentralNucleiAxesL              matlab.ui.control.UIAxes
        CentralNucleiAxesR              matlab.ui.control.UIAxes
        NonfiberPanel                   matlab.ui.container.Panel
        NonfiberThresholdLabel          matlab.ui.control.Label
        NonfiberAccept                  matlab.ui.control.Button
        NonfiberAdjust                  matlab.ui.control.Button
        NonfiberThreshold               matlab.ui.control.NumericEditField
        ThresholdEditField_2Label_2     matlab.ui.control.Label
        NonfiberAxesL                   matlab.ui.control.UIAxes
        NonfiberAxesR                   matlab.ui.control.UIAxes
        NonfiberClassificationControlPanel  matlab.ui.container.Panel
        NonfiberClassificationDescription_2  matlab.ui.control.Label
        NonfiberClassificationDescription  matlab.ui.control.Label
        NonfiberClassificationFileWriteStatusLabel  matlab.ui.control.Label
        NonfiberClassificationColorDropDown  matlab.ui.control.DropDown
        ClassificationColorLabel        matlab.ui.control.Label
        DoneNonfiberClassification      matlab.ui.control.Button
        WritetoExcelNonfiberClassification  matlab.ui.control.Button
        ClassifyNonfiberObjects         matlab.ui.control.Button
        NonfiberClassificationDataOutputFolder  matlab.ui.control.EditField
        DataOutputFolderEditField_3Label_3  matlab.ui.control.Label
        PixelSizeNonfiberClassification  matlab.ui.control.NumericEditField
        PixelSizeumpixelEditField_3Label_2  matlab.ui.control.Label
        NonfiberClassificationChannelColorBox  matlab.ui.control.UIAxes
        CentralNucleiControlPanel       matlab.ui.container.Panel
        CentralNucleiFileWriteStatusLabel  matlab.ui.control.Label
        CentralNucleiDescription_2      matlab.ui.control.Label
        CentralNucleiDescription        matlab.ui.control.Label
        CentralNucleiDataOutputFolder   matlab.ui.control.EditField
        DataOutputFolderEditField_2Label  matlab.ui.control.Label
        MinimumNucleusSizeum2EditField  matlab.ui.control.NumericEditField
        MinimumNucleusSizeum2EditFieldLabel  matlab.ui.control.Label
        DistancefromborderEditField     matlab.ui.control.NumericEditField
        DistancefromborderEditFieldLabel  matlab.ui.control.Label
        DoneCentralNuclei               matlab.ui.control.Button
        CentralNucleiExcelWrite         matlab.ui.control.Button
        CalculateCentralNuclei          matlab.ui.control.Button
        NucleiColorDropDown             matlab.ui.control.DropDown
        NucleiColorDropDownLabel        matlab.ui.control.Label
        PixelSizeCentralNuclei          matlab.ui.control.NumericEditField
        PixelSizeumpixelLabel           matlab.ui.control.Label
        CentralNucleiChannelColorBox    matlab.ui.control.UIAxes
        NonfiberObjectsControlPanel     matlab.ui.container.Panel
        NonfiberObjectsDescription_2    matlab.ui.control.Label
        NonfiberObjectsDescription      matlab.ui.control.Label
        NonfiberObjectsFileWriteStatusLabel  matlab.ui.control.Label
        PixelSizeNonfiberObjects        matlab.ui.control.NumericEditField
        PixelSizeumpixelLabel_2         matlab.ui.control.Label
        DoneNonfiber                    matlab.ui.control.Button
        WritetoExcelNonfiber            matlab.ui.control.Button
        CalculateNonfiberObjects        matlab.ui.control.Button
        NonfiberObjectsDataOutputFolder  matlab.ui.control.EditField
        DataOutputFolderEditField_3Label_2  matlab.ui.control.Label
        NonfiberObjectsColorDropDown    matlab.ui.control.DropDown
        ObjectColorDropDownLabel        matlab.ui.control.Label
        NonfiberChannelColorBox         matlab.ui.control.UIAxes
        FiberPropertiesControlPanel     matlab.ui.container.Panel
        FiberPropertiesFileWriteStatusLabel  matlab.ui.control.Label
        FiberPropertiesDescription_2    matlab.ui.control.Label
        FiberPropertiesDescription      matlab.ui.control.Label
        CalculateFiberProperties        matlab.ui.control.Button
        PixelSizeFiberProperties        matlab.ui.control.NumericEditField
        PixelSizeumpixelEditFieldLabel_2  matlab.ui.control.Label
        FiberPropertiesDataOutputFolder  matlab.ui.control.EditField
        DataOutputFolderEditFieldLabel  matlab.ui.control.Label
        DoneFiberProperties             matlab.ui.control.Button
        WritetoExcelButton              matlab.ui.control.Button
        ManualFilterControls            matlab.ui.container.Panel
        ManualFilterDescription_2       matlab.ui.control.Label
        ManualFilterDescription         matlab.ui.control.Label
        FinishManualFilteringButton     matlab.ui.control.Button
        RemoveNonfibersButton           matlab.ui.control.Button
        SegmentationParameters          matlab.ui.container.Panel
        CloseInitialSegmentationButton  matlab.ui.control.Button
        InitialSegmentationDescription_2  matlab.ui.control.Label
        InitialSegmentationDirections   matlab.ui.control.Label
        InitialSegmentationDescription  matlab.ui.control.Label
        DetectValueButton               matlab.ui.control.Button
        SegmentationThresholdSlider     matlab.ui.control.Slider
        SegmentationThresholdSliderLabel  matlab.ui.control.Label
        AcceptSegmentationButton        matlab.ui.control.Button
        FiberOutlineColorDropDown       matlab.ui.control.DropDown
        FiberOutlineColorDropDownLabel  matlab.ui.control.Label
        SegmentButton                   matlab.ui.control.Button
        FiberOutlineChannelColorBox     matlab.ui.control.UIAxes
        FiberPredictionControlPanel     matlab.ui.container.Panel
        FiberPredictionDescription_3    matlab.ui.control.Label
        FiberPredictionDescription_2    matlab.ui.control.Label
        FiberPredictionDescription      matlab.ui.control.Label
        PixelSizeFiberPrediction        matlab.ui.control.NumericEditField
        PizelSizeumpixelLabel           matlab.ui.control.Label
        SortingThresholdSlider          matlab.ui.control.Slider
        SortingThresholdHigherrequiresmoremanualsortingLabel  matlab.ui.control.Label
        ManualSortingButton             matlab.ui.control.Button
        FilterButton                    matlab.ui.control.Button
        ManualSegmentationControls      matlab.ui.container.Panel
        ManualSegmentationDescription_4  matlab.ui.control.Label
        ManualSegmentationDescription_3  matlab.ui.control.Label
        ManualSegmentationDescription_2  matlab.ui.control.Label
        ManualSegmentationDescription   matlab.ui.control.Label
        FinishDrawingButton             matlab.ui.control.Button
        StartMergingButton              matlab.ui.control.Button
        MergeObjectsModeLabel           matlab.ui.control.Label
        DrawingModeLabel                matlab.ui.control.Label
        CloseManualSegmentationButton   matlab.ui.control.Button
        AcceptLineButton                matlab.ui.control.Button
        StartDrawingButton              matlab.ui.control.Button
        SelectFileDescription_2         matlab.ui.control.Label
        SelectFileDescription           matlab.ui.control.Label
        ImageBackground                 matlab.ui.container.Panel
        UIAxes                          matlab.ui.control.UIAxes
        BatchModeLabel                  matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        Panel                           matlab.ui.container.Panel
        Hyperlink_2                     matlab.ui.control.Hyperlink
        Hyperlink                       matlab.ui.control.Hyperlink
        Panel_2                         matlab.ui.container.Panel
        Image2                          matlab.ui.control.Image
        SMASHLabel                      matlab.ui.control.Label
        Toolbar                         matlab.ui.container.Panel
        NonfiberClassificationButton    matlab.ui.control.Button
        InitialSegmentationButton       matlab.ui.control.Button
        NonfiberObjectsButton           matlab.ui.control.Button
        FiberTypingButton               matlab.ui.control.Button
        CentralNucleiButton             matlab.ui.control.Button
        FiberPropertiesButton           matlab.ui.control.Button
        ManualFiberFilterButton         matlab.ui.control.Button
        FiberPredictionButton           matlab.ui.control.Button
        ManualSegmentationButton        matlab.ui.control.Button
        SortingAxesPanel                matlab.ui.container.Panel
        MarkasfiberLabel                matlab.ui.control.Label
        NoButton                        matlab.ui.control.Button
        YesButton                       matlab.ui.control.Button
        SortingAxesR                    matlab.ui.control.UIAxes
        SortingAxesL                    matlab.ui.control.UIAxes
        Prompt                          matlab.ui.control.Label
        FilenameLabel                   matlab.ui.control.Label
        SelectFilesButton               matlab.ui.control.Button
        SelectFileErrorLabel            matlab.ui.control.Label
    end

    
    properties (Access = private)
        default
        Files
        orig_img
        orig_img_multispectral
        bw_obj
        model
        pix_size
        notfiber
        confirm
        output_path
        done
        num_obj
        props
        thresh_nuc
        cen_pix
        cen_nuc
        fprop
        FT_Adj
        cutoff_avg
        ave_g
        ponf
        areas
        CentralNucleiAdj
        Obj_Adj
        thresh_nf
        num_nf
        nf_data
        nf_mask
        nf_bw_obj

        % Nonfiber classification properties
        nonfiber_classification_cutoff_avg
        classified_nonfiber_num_obj
        classified_nonfiber_ave_g
        classified_nonfiber_ponf
        classified_nonfiber_areas
        classified_nonfiber_percentage
        
        segmodel
        channelRGB % Contains all the RGB values of the channel colors
        IsBatchMode % Boolean to store whether running in batch mode
        BatchModeFileNames % Store batch mode file names
        BatchModePathName % Store batch mode path name
        BatchModeFilterIndex % Store batch mode filter index
        AcceptedFileExtensions
    end
    
    methods (Access = private)

        function FileInitialization(app, FileName, PathName, FilterIndex)
            drawnow limitrate;
            figure(app.UIFigure)

            % Reset app menu bar
            DisableMenuBarButtonsAndClearFileLabels(app);
            
            if FilterIndex
                if FileName == 0
                    return
                end
                C = strsplit(FileName,'.');
                ExtName = C(end);
                FileNameS = (C(1:(end-1)));
                FileNameS = FileNameS{1};
                MaskName = strcat(FileNameS,'_mask.',ExtName);
                MaskName = MaskName{1};
                NFMaskName = strcat(FileNameS,'_nf_mask.',ExtName);
                NFMaskName = NFMaskName{1};
            end
            
            if app.IsBatchMode == 1
                numberOfFilesForBatchMode = length(app.BatchModeFileNames);
                batchModeLabelString = "Batch Mode (" + int2str(numberOfFilesForBatchMode) + " files)";
                app.BatchModeLabel.Text = strcat(batchModeLabelString);
                app.BatchModeLabel.Visible = 'on';
            else
                app.BatchModeLabel.Visible = 'off';
            end

            app.FilenameLabel.Text = FileNameS;
            app.Files{1} = FileName;
            app.Files{2} = MaskName;
            app.Files{3} = PathName;
            app.Files{4} = FileNameS;
            app.Files{5} = NFMaskName;
            cd(PathName)
    
            % Change output directory to where image is located
            app.FiberPropertiesDataOutputFolder.Value = pwd;
            app.CentralNucleiDataOutputFolder.Value = pwd;
            app.FiberTypingDataOutputFolder.Value = pwd;
            app.NonfiberObjectsDataOutputFolder.Value = pwd;
            app.NonfiberClassificationDataOutputFolder.Value = pwd;
            
            BioformatsData = bfopen(FileName);
            PixelDataForAllLayers = BioformatsData{1,1};
            ColorMapDataForAllLayers = BioformatsData{1, 3};

            TotalColorDropDownItems = {};
            TotalColorDropDownItemsData = {};

            if isempty(ColorMapDataForAllLayers{1,1})
                % BioFormats does not assign a color map for RGB images.
                % To keep code for multilayer and RGB case consistent, we
                % create our own colormap for the RGB case.
                ColorMapDataForAllLayers = app.GetDefaultColorMap(class(PixelDataForAllLayers{1,1}));
            end

            LayerOnePixelData = PixelDataForAllLayers{1,1};
            LayerSize = size(LayerOnePixelData);
            RGBSize = [LayerSize 3];

            TotalRGB = zeros(RGBSize, 'uint8');
            TotalMultiSpectral = [];

            
            % Channel RGB takes in all RGB values for the 
            % different colors in the channels
            app.channelRGB = [];

            NumLayers = length(PixelDataForAllLayers);
            for Layer = 1:NumLayers
                PixelsGrayscale = PixelDataForAllLayers{Layer, 1};
                ColorMap = ColorMapDataForAllLayers{1, Layer};
                PixelsRGBAsDouble = ind2rgb(PixelsGrayscale, ColorMap);
                PixelsRGBAsUInt8 = im2uint8(PixelsRGBAsDouble);
                PixelsGrayscaleUInt8 = im2uint8(PixelsGrayscale);

                % Retrieve color name from the RGB values
                RGBValues = ColorMap(end,:);
                ConvertedRGB = colornames('MATLAB', RGBValues,'RGB');
                ColorName = char(ConvertedRGB);

                % Autoscaling - scale the pixel intensity for each channel
                MaxIntensity = max(PixelsGrayscaleUInt8,[],'all');
                MinIntensity = min(PixelsGrayscaleUInt8,[],'all');
                ActualRange = MaxIntensity - MinIntensity;
                UInt8Range = 255;
                ScalingFactor = UInt8Range / ActualRange;
                PixelsGrayscaleUInt8 = ScalingFactor * PixelsGrayscaleUInt8;
                PixelsRGBAsUInt8 = ScalingFactor * PixelsRGBAsUInt8;

                TotalMultiSpectral = cat(3, TotalMultiSpectral, PixelsGrayscaleUInt8);
                TotalRGB = imadd(TotalRGB, PixelsRGBAsUInt8);

                % Include only color name in the channel drop down menu
                TotalColorDropDownItems = cat(2, TotalColorDropDownItems, ColorName);
                TotalColorDropDownItemsData  = cat(2, TotalColorDropDownItemsData, {num2str(Layer)});

                % Add the RGB values of the channel color
                % Convert from cell array to double matrix when
                % appending
                app.channelRGB = cat(1, app.channelRGB, RGBValues(1,:));
            end


            app.orig_img = TotalRGB;
            app.orig_img_multispectral = TotalMultiSpectral;


            app.FiberOutlineColorDropDown.Items = TotalColorDropDownItems;
            app.FiberOutlineColorDropDown.ItemsData = TotalColorDropDownItemsData;
            app.NucleiColorDropDown.Items = TotalColorDropDownItems;
            app.NucleiColorDropDown.ItemsData = TotalColorDropDownItemsData;
            app.FiberTypeColorDropDown.Items = TotalColorDropDownItems;
            app.FiberTypeColorDropDown.ItemsData = TotalColorDropDownItemsData;
            app.NonfiberObjectsColorDropDown.Items = TotalColorDropDownItems;
            app.NonfiberObjectsColorDropDown.ItemsData = TotalColorDropDownItemsData;
            app.NonfiberClassificationColorDropDown.Items = TotalColorDropDownItems;
            app.NonfiberClassificationColorDropDown.ItemsData = TotalColorDropDownItemsData;

            % Display the image
            imshow(app.orig_img,'Parent',app.UIAxes);

            % Enable the correct menu bar buttons
            if ~exist(MaskName,'file')
                % If no mask exists, the image first needs to be segmented
                app.InitialSegmentationButton.Enable = 'on';
            elseif app.IsBatchMode == 0
                % If mask exists and batch mode is not enabled, enable all
                % buttons
                EnableMenuBarButtons(app);
            else
                % In batch mode, only allow initial segmentation and fiber
                % prediction
                app.InitialSegmentationButton.Enable = 'on';
                app.FiberPredictionButton.Enable = 'on';
            end
            
            UpdateColorChannelBox(app)

            % Allow user to select different files.
            app.SelectFilesButton.Enable = 'on';
        end

        function UpdateColorChannelBox(app)
            % Color Channel Box   
            % Get RGB values of color channel and convert to numeric matrix
            % Update color on color box with appropriate RGB channel

            % Nonfiber objects
            NonfiberRGBChannel = app.channelRGB(str2double(app.NonfiberObjectsColorDropDown.Value), :);
            app.NonfiberChannelColorBox.Color = NonfiberRGBChannel;

            % Central Nuclei
            CentralNucleiRGBChannel = app.channelRGB(str2double(app.NucleiColorDropDown.Value), :);
            app.CentralNucleiChannelColorBox.Color = CentralNucleiRGBChannel;

            % Fiber Typing
            FiberTypingChannel = app.channelRGB(str2double(app.FiberTypeColorDropDown.Value), :);
            app.FiberTypingChannelColorBox.Color = FiberTypingChannel;

            % Fiber Outline
            FiberOutlineChannel = app.channelRGB(str2double(app.FiberOutlineColorDropDown.Value), :);
            app.FiberOutlineChannelColorBox.Color = FiberOutlineChannel;
            
            % Nonfiber objects classification
            NonfiberClassificationRGBChannel = app.channelRGB(str2double(app.NonfiberClassificationColorDropDown.Value), :);
            app.NonfiberClassificationChannelColorBox.Color = NonfiberClassificationRGBChannel;
        end

        function SegmentAndDisplayImage(app)
            foc = app.FiberOutlineColorDropDown.Value;
            foc = str2double(foc);
            lam = app.orig_img_multispectral(:,:,foc);  % fiber outline color
            
            % Image Segmentation
            lam_t = imhmin(lam,app.SegmentationThresholdSlider.Value);
            WS = watershed(lam_t);
            app.bw_obj = imcomplement(logical(WS));
            
            % Display segmented image
            flat_img =flattenMaskOverlay(app.orig_img,app.bw_obj,1,'w');
            imshow(flat_img,'Parent',app.UIAxes);
        end
                   
        function AquireMaskFiberPrediction(app)
            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
            app.bw_obj = imclearborder(app.bw_obj,4);
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
        end

        function FiberPredictionWithMediumTree(app)
            app.FilterButton.Enable = 'off';
            app.SortingThresholdSlider.Enable = 'off';
            app.Prompt.Text = 'Filtering, please wait.';
            
            drawnow limitrate
            pix_area = app.pix_size^2;
            label = bwlabel(app.bw_obj,4);
            %num_obj = max(max(label));
            
            % Get properties of regions
            Rprop = regionprops('table',label,'Centroid','Area','Eccentricity','Solidity','Extent','Circularity','PixelIdxList');
            predictors = table(Rprop.Area,Rprop.Eccentricity,Rprop.Solidity,Rprop.Extent,Rprop.Circularity);
            predictors.Properties.VariableNames{1} = 'Area';
            predictors.Properties.VariableNames{2} = 'Eccentricity';
            predictors.Properties.VariableNames{3} = 'Convexity';
            predictors.Properties.VariableNames{4} = 'Extent';
            predictors.Properties.VariableNames{5} = 'Circularity';
           
            % Predict if a region is a fiber
            classifier = app.model.MediumTree.ClassificationTree;
            predictors.Area = pix_area*predictors.Area;
            [Class, Score] = predict(classifier,predictors);
            fiberindex = find(Class == 'Fiber');
            nonfiberindex = find(Class == 'Nonfiber');
            fiberfiltered = ismember(label,fiberindex);
            nonfiberfiltered = ismember(label,nonfiberindex);
            
            % Determine if fiber prediction is a "maybe"
            maybe = abs(Score(:,1)-Score(:,2));
            maybe = find(maybe < app.SortingThresholdSlider.Value);  % Change this value to adjust sensitivty
            maybefiltered = ismember(label,maybe);
            
            tempmask = label;
            dispmask = logical((nonfiberfiltered.*maybefiltered) + fiberfiltered); % Show only the fibers and nonfiber maybes
            
            imshow(flattenMaskOverlay(app.orig_img,dispmask,0.5,'w'),'Parent',app.UIAxes);
            app.Prompt.Text = '';

            SaveMaskToMaskFile(app, tempmask);
        end
                   
        function CreateFolderIfDirectoryIsNonexistent(~, pathDirectory)
            if exist(pathDirectory,'dir') == 0
                mkdir(pathDirectory)
            end
        end

        function SaveMaskToMaskFile(app, mask)
            rgb_label = label2rgb(mask,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff')
        end

        function results = IsROIPositionInBound(app, xp, yp)
            xp_is_valid = xp > 0 & xp <= size(app.bw_obj,2);
            yp_is_valid = yp > 0 & yp <= size(app.bw_obj,1);
            results = xp_is_valid && yp_is_valid;
        end

        function results = ReadMaskFromMaskFile(app)
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            results = imbinarize(graymask,0.99);
        end

        function SaveNFMaskToMaskFile(app, mask)
            imwrite(mask,app.Files{5},'tiff');
        end

        function results = ReadNFMaskFromMaskFile(app)
            results = imread(app.Files{5});
        end

        %% ==================== Merge Region Functions ====================

        function [new_bw_obj, is_merge_successful] = MergeObjects(app, label, first_region_to_merge, second_region_to_merge)
            % The strategy for merging is to take the two regions and
            % expand each region one pixel in each direction. Wherever the
            % newly expanded regions overlap are potential pixels we can
            % fill in to merge the regions. However, there is an edge case
            % where if a foreign region is also touching one of these
            % overlap pixels, we want to avoid filling in these pixels
            % because they would cause the foreign region to also become
            % merged with the selected regions.

            % Expand regions in each direction.
            padded_first_region = PadBWOnePixelInEachDirection(app, label == first_region_to_merge);
            padded_second_region = PadBWOnePixelInEachDirection(app, label == second_region_to_merge);

            % Find overlap.
            potential_pixels_to_fill_in = padded_first_region & padded_second_region;
            is_merge_successful = any(any(potential_pixels_to_fill_in));

            % Filter out pixels that are touching foreign regions.
            bw_pixels_required_to_merge = FilterPointsThatWouldCauseUndesiredMerge(app, potential_pixels_to_fill_in, label, first_region_to_merge, second_region_to_merge);

            % Return result.
            new_bw_obj = app.bw_obj;
            new_bw_obj(bw_pixels_required_to_merge == 1) = 1;
        end

        function results = PadBWOnePixelInEachDirection(app, bw)
            % Create shifted masks for each direction
            left = ShiftLeft(app, bw);
            right = ShiftRight(app, bw);
            up = ShiftUp(app, bw);
            down = ShiftDown(app, bw);
            upleft = ShiftUp(app, left);
            upright = ShiftUp(app, right);
            downleft = ShiftDown(app, left);
            downright = ShiftDown(app, right);

            % OR the masks together to pad bw
            results = bw | up | down | left | right | upleft | upright | downleft | downright;
        end

        function results = ShiftLeft(~, mat)
            results = circshift(mat,[0 -1]);
            % Remove wrapped around pixels
            results(:,end) = 0;
        end

        function results = ShiftRight(~, mat)
            results = circshift(mat,[0 1]);
            % Remove wrapped around pixels
            results(:,1) = 0;
        end

        function results = ShiftUp(~, mat)
            results = circshift(mat,[-1 0]);
            % Remove wrapped around pixels
            results(end,:) = 0;
        end

        function results = ShiftDown(~, mat)
            results = circshift(mat,[1 0]);
            % Remove wrapped around pixels
            results(1,:) = 0;
        end

        function results = FilterPointsThatWouldCauseUndesiredMerge(app, bw_boundary_intersection, object_labels, first_region_to_merge, second_region_to_merge)
            [row, col] = find(bw_boundary_intersection == 1);
            results = bw_boundary_intersection;
            for i = 1:numel(row)
                r = row(i);
                c = col(i);
                if ~IsPointAdjacentToAcceptableLabels(app, r, c, object_labels, first_region_to_merge, second_region_to_merge)
                    results(r,c) = 0;
                end
            end
        end

        function results = IsPointAdjacentToAcceptableLabels(~, r, c, object_labels, first_region_to_merge, second_region_to_merge)
            point_neighbor_labels = [object_labels(r+1,c), object_labels(r-1,c), object_labels(r,c+1), object_labels(r,c-1)];
            % A neighbor is acceptable as long as it's either a non-object (0) or one of two objects being merge.
            acceptable_labels = [0 first_region_to_merge second_region_to_merge];
            results = all(ismember(point_neighbor_labels, acceptable_labels));
        end

        function EnableMenuBarButtons(app)
            app.SelectFilesButton.Enable = 'on';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';

            % Only allow nonfiber classification if non-fiber objects mask
            % exists
            if exist(app.Files{5},'file') 
                app.NonfiberClassificationButton.Enable = 'on';
            end

            app.SelectFileDescription.Visible = 'on';
            app.SelectFileDescription_2.Visible = 'on';
        end
        
        function DisableMenuBarButtonsAndClearFileLabels(app)
            app.SelectFilesButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.NonfiberClassificationButton.Enable = 'off';
            app.SelectFileErrorLabel.Visible = 'off';
            app.SelectFileDescription.Visible = 'off';
            app.SelectFileDescription_2.Visible = 'off';
        end

        function SyncPixelSize(app)
            app.PixelSizeFiberProperties.Value = app.pix_size;
            app.PixelSizeCentralNuclei.Value = app.pix_size;
            app.PixelSizeFiberTyping.Value = app.pix_size;
            app.PixelSizeNonfiberObjects.Value = app.pix_size;
            app.PixelSizeNonfiberClassification.Value = app.pix_size;
            app.PixelSizeFiberPrediction.Value = app.pix_size;
        end

        function filter = AcceptedFileExtensionsFilter(app)
        % AcceptedFileExtensionsFilter  Returns file type filter
        %   AcceptedFileExtensionsFilter(app) returns a cell array of filters to be
        %   used by uigetfile.
        %
        %   See also uigetfile.

            % This is needed because uigetfile filter requires each filter
            % to be on a separate row. In our case, this means reshaping
            % the array to have one column.
            filter = strcat('*.', app.AcceptedFileExtensions);
        end

        function isAccepted = IsFileAccepted(app, filename)
            fileType = GetFileExtension(app, filename);
            isAccepted = ismember(fileType, app.AcceptedFileExtensions);
        end

        function fileExtension = GetFileExtension(~, filename)
            C = strsplit(filename,'.');
            fileExtension = C{end};
        end

        function message = GetFileExtensionErrorMessage(app, filename)
            acceptedFileTypeString = join(app.AcceptedFileExtensions, ', ');
            message = [filename, ' does not have one of the appropriate file extensions: ', acceptedFileTypeString{1}];
        end

        function results = GetDefaultColorMap(~, classname)
            % GetDefaultColorMap  Returns a Bio-Formats formatted RGB map
            % for RGB case
            %
            %  Example return value:
            %
            %   3×3 cell array
            %
            %    {255×3 single}    {255×3 single}    {255×3 single}
            %    {  0×0 double}    {  0×0 double}    {  0×0 double}
            %    {  0×0 double}    {  0×0 double}    {  0×0 double}

            allNumsColumn = reshape(1:intmax(classname),[], 1);
            allNumsColumnScaled = single(double(allNumsColumn) / double(intmax(classname)));
            allZerosColumn = zeros(size(allNumsColumnScaled, 1), 1);
            redMap = [allNumsColumnScaled allZerosColumn allZerosColumn];
            greenMap = [allZerosColumn allNumsColumnScaled allZerosColumn];
            blueMap = [allZerosColumn allZerosColumn allNumsColumnScaled];

            results = cell(3,3);
            results(1,:) = {redMap, greenMap, blueMap};
        end
    end


    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.default = readcell('SMASH Defaults.xlsx');
            app.model = load('MediumTreeModel.mat','MediumTree');
            app.segmodel = load('SegmentationModel.mat','segModel');
            % Apply defaults
            app.pix_size = app.default{2,2};
            SyncPixelSize(app);
            %app.SegmentationThresholdSlider.Value = app.default{7,2};
            app.FiberOutlineColorDropDown.Value = num2str(app.default{3,2});
            app.NucleiColorDropDown.Value = num2str(app.default{4,2});
            app.DistancefromborderEditField.Value = app.default{14,2};
            app.MinimumNucleusSizeum2EditField.Value = app.default{15,2};
            app.FiberTypeColorDropDown.Value = num2str(app.default{5,2});
            app.NonfiberObjectsColorDropDown.Value = num2str(app.default{6,2});
            app.AcceptedFileExtensions = {'tif';'tiff';'lif';'jpg';'png';'bmp';'czi'};
            % TODO - default for nofiber classification
            linkaxes([app.SortingAxesL, app.SortingAxesR]);
        end

        % Button pushed function: SelectFilesButton
        function SelectFilesButtonPushed(app, event)
            % Allow user to select multiple files
            if ismac()
                [FileNames,PathName,FilterIndex] = uigetfile("*",'File Selector - dont select mask', 'MultiSelect','on');
            else
                [FileNames,PathName,FilterIndex] = uigetfile(app.AcceptedFileExtensionsFilter(),'File Selector - dont select mask', 'MultiSelect','on');
            end

            % Set pathname and filter index, 
            app.BatchModePathName = PathName;
            app.BatchModeFilterIndex = FilterIndex;
            app.BatchModeFileNames = FileNames;

            % Return if there no filenames
            if ~FilterIndex
                return
            end

            if iscell(FileNames) % If there are multiple FileNames, batch mode
                app.IsBatchMode = 1;
                FileName = FileNames{1};
            else
                app.IsBatchMode = 0;
                FileName = FileNames;
            end

            if ~IsFileAccepted(app, FileName)
                app.SelectFileErrorLabel.Text = GetFileExtensionErrorMessage(app, FileName);
                app.SelectFileErrorLabel.Visible = 'on';
                return
            end

            app.SelectFileErrorLabel.Visible = 'off';

            % Run file initialization
            FileInitialization(app, FileName, PathName, FilterIndex);
            app.ImageBackground.Visible = 'on';
        end

        % Button pushed function: SegmentButton
        function SegmentButtonPushed(app, event)
            go = 1;
               files = dir;
               
               % Warn the user if a mask file already exists
               if find(strcmp({files.name},app.Files{2}),1) > 0
                   warn = uiconfirm(app.UIFigure, 'This will delete any existing segmentation or fiber prediction.', 'Confirm Segmentation','Icon','Warning');
                   warn = convertCharsToStrings(warn);
                   if strcmp(warn,'Cancel')
                       go = 0;
                   end
               end
               
               if go
                   %orig_img = imread(app.Files{1});
                   SegmentAndDisplayImage(app);
                   app.AcceptSegmentationButton.Enable = 'on';
                   
               end
        end

        % Button pushed function: StartDrawingButton
        function StartDrawingButtonPushed(app, event)
            app.done = 0;
            app.CloseManualSegmentationButton.Enable = 'off';
            app.StartDrawingButton.Enable = 'off';
            app.FinishDrawingButton.Enable = 'on';
            app.StartMergingButton.Enable = 'off';
            
            while true
                app.AcceptLineButton.Enable = 'on';
                app.Prompt.Text = 'Draw line to separate fibers and adjust as needed. Right click line to delete.';
                h = drawfreehand(app.UIAxes,'Closed',false,'FaceAlpha',0);
                uiwait(app.UIFigure);
                if app.done
		            break;
                end
                app.AcceptLineButton.Enable = 'off';
                if isvalid(h)   % Checks if line exists or was deleted
                    mask = createMask(h);
                    mask2 = bwmorph(mask,'bridge');
                    app.bw_obj = logical(app.bw_obj + mask2);
                    flat_img = flattenMaskOverlay(app.orig_img, app.bw_obj, 1, 'w');
                    imshow(flat_img,'Parent',app.UIAxes);
                end 
                uiresume(app.UIFigure);
            end
        end

        % Button pushed function: AcceptLineButton
        function AcceptLineButtonPushed(app, event)
            uiresume(app.UIFigure);
            app.Prompt.Text = '';
        end

        % Button pushed function: CloseManualSegmentationButton
        function CloseManualSegmentationButtonPushed(app, event)
            app.ManualSegmentationControls.Visible = 'off';
            EnableMenuBarButtons(app);
            app.SegmentationParameters.Visible = 'off';
            app.Prompt.Text = '';
        end

        % Button pushed function: FilterButton
        function FilterButtonPushed(app, event)
            
            if app.IsBatchMode == 0
                app.FilterButton.Enable = 'off';
                app.SortingThresholdSlider.Enable = 'off';
                app.Prompt.Text = 'Filtering, please wait.';
                
                drawnow limitrate
                pix_area = app.pix_size^2;
                label = bwlabel(app.bw_obj,4);
                %num_obj = max(max(label));
            
                % Get properties of regions
                Rprop = regionprops('table',label,'Centroid','Area','Eccentricity','Solidity','Extent','Circularity','PixelIdxList');
                cents = cat(1,Rprop.Centroid);
                predictors = table(Rprop.Area,Rprop.Eccentricity,Rprop.Solidity,Rprop.Extent,Rprop.Circularity);
                predictors.Properties.VariableNames{1} = 'Area';
                predictors.Properties.VariableNames{2} = 'Eccentricity';
                predictors.Properties.VariableNames{3} = 'Convexity';
                predictors.Properties.VariableNames{4} = 'Extent';
                predictors.Properties.VariableNames{5} = 'Circularity';
           
                % Predict if a region is a fiber
                classifier = app.model.MediumTree.ClassificationTree;
                predictors.Area = pix_area*predictors.Area;
                [Class, Score] = predict(classifier,predictors);
                fiberindex = find(Class == 'Fiber');
                nonfiberindex = find(Class == 'Nonfiber');
                fiberfiltered = ismember(label,fiberindex);
                nonfiberfiltered = ismember(label,nonfiberindex);
            
                % Determine if fiber prediction is a "maybe"
                maybe = abs(Score(:,1)-Score(:,2));
                maybe = find(maybe < app.SortingThresholdSlider.Value);  % Change this value to adjust sensitivty
                maybefiltered = ismember(label,maybe);
                
                tempmask = label;
            
            
                codedim = cat(3,nonfiberfiltered,fiberfiltered,maybefiltered); % Color coded image
                codedim = uint8(codedim);                                      % Magenta - nonfibers
                codedim = 255.*codedim;                                        % Cyan - Fibers
            
                dispmask = logical((nonfiberfiltered.*maybefiltered) + fiberfiltered); % Show only the fibers and nonfiber maybes
            
                imshow(flattenMaskOverlay(app.orig_img,dispmask,0.5,'w'),'Parent',app.UIAxes);
                app.Prompt.Text = '';
                app.ManualSortingButton.Enable = 'on';
                uiwait(app.UIFigure);
            
            
                    app.SortingAxesPanel.Visible = 'on';
                    
                    % Manual sorting
                    
                    for i = 1:length(maybe)
                        app.SortingAxesPanel.Title = [num2str(i) ' of ' num2str(length(maybe))];
                        app.notfiber = 0;
                        imshow(codedim,'Parent',app.SortingAxesL);
                        hold(app.SortingAxesL,'on');
                        plot(cents(maybe(i),1),cents(maybe(i),2),'y*','Parent',app.SortingAxesL)
                        xlim(app.SortingAxesL,[(cents(maybe(i),1)-100) (cents(maybe(i),1)+100)])
                        ylim(app.SortingAxesL,[(cents(maybe(i),2)-100) (cents(maybe(i),2)+100)])
                        
                    
                        imshow(app.orig_img,'Parent',app.SortingAxesR);
                        hold(app.SortingAxesR,'on');
                        plot(cents(maybe(i),1),cents(maybe(i),2),'y*','Parent',app.SortingAxesR)
                        xlim(app.SortingAxesR,[(cents(maybe(i),1)-100) (cents(maybe(i),1)+100)])
                        ylim(app.SortingAxesR,[(cents(maybe(i),2)-100) (cents(maybe(i),2)+100)])
                    
                        uiwait(app.UIFigure);
                        
                        if app.notfiber
                            removeidx = Rprop(maybe(i),7);
                            tempmask(removeidx.PixelIdxList{1,1}) = 0;
                        end
                        hold(app.SortingAxesL,'off');
                        hold(app.SortingAxesR,'off');
                    end
                


                app.SortingAxesPanel.Visible = 'off';
                app.ImageBackground.Visible = 'on';
            
            hiprobnon = ~ismember(nonfiberindex,maybe);  % Find the index of regions that are nonfiber with high probability
            hiprobnonidx = nonzeros(hiprobnon.*nonfiberindex);
            for i = 1:length(hiprobnonidx)
               hiprobnonprop = Rprop(hiprobnonidx(i),7);
               tempmask(hiprobnonprop.PixelIdxList{1,1}) = 0;
            end
            
            imshow(flattenMaskOverlay(app.orig_img,logical(tempmask),0.5,'w'),'Parent',app.UIAxes);

            SaveMaskToMaskFile(app, tempmask);

            app.FiberPredictionControlPanel.Visible = 'off';
            EnableMenuBarButtons(app);

            else
                app.SortingThresholdSlider.Enable = 'off';
                numberOfFilesSelected = length(app.BatchModeFileNames);

                for k=1:numberOfFilesSelected
                    % Retrieve the filename of the current file
                    currentFile = app.BatchModeFileNames{k};
                    % Check if the file type is correct, if not skip
                    if ~IsFileAccepted(app, currentFile)
                        % Add the prompt for warning that file type is not
                        % acceped
                        app.Prompt.FontColor = [1 1 0];
                        promptString = "<Batch Mode - Warning: " + currentFile + " skipped because it is not a valid file type>";
                        app.Prompt.Text = promptString;
                    else
                        % Add the prompt at the top for indication that
                        % batch mode is running on which file
                        promptString = "Batch Mode - In Progress: " + int2str(k-1) + " out of " + int2str(numberOfFilesSelected) + " completed.";
                        app.Prompt.Text = promptString;
                        % Go through file initilization
                        % In file initialization, it enables all the other buttons,
                        % so you need to change this button to the initial page and
                        % then continue
                        FileInitialization(app, currentFile, app.BatchModePathName, app.BatchModeFilterIndex);
                        % Run filter button functionality
                        AquireMaskFiberPrediction(app);
                        FiberPredictionWithMediumTree(app);

                    end
                end
                app.Prompt.FontColor = [0 0 0];
                app.Prompt.Text = 'Batch Mode - Fiber Prediction Completed.';
            end


            
        end

        % Button pushed function: AcceptSegmentationButton
        function AcceptSegmentationButtonPushed(app, event)
            if app.IsBatchMode == 0
                label = bwlabel(~logical(app.bw_obj),4);
                SaveMaskToMaskFile(app, label);
                app.SegmentationParameters.Visible = 'off';
                EnableMenuBarButtons(app);
            else 
                numberOfFilesSelected = length(app.BatchModeFileNames);

                for k=1:numberOfFilesSelected
                    % Retrieve the filename of the current file
                    currentFile = app.BatchModeFileNames{k};
                    % Check if the file type is correct, if not skip
                    if ~IsFileAccepted(app, currentFile)
                        % Add the prompt for warning that file type is not
                        % acceped
                        promptString = "<Batch Mode - Warning: " + currentFile + " skipped because it is not a valid file type>";
                        app.Prompt.FontColor = [1 1 0];
                        app.Prompt.Text = promptString;
                    else
                        % Add the prompt at the top for indication that batch
                        % mode is running on which file
                        promptString = "Batch Mode - In Progress: " + int2str(k-1) + " out of " + int2str(numberOfFilesSelected) + " completed.";
                        app.Prompt.Text = promptString;
                        % Go through file initilization
                        % In file initialization, it enables all the other buttons,
                        % so you need to change this button to the initial page and
                        % then continue
                        FileInitialization(app, currentFile, app.BatchModePathName, app.BatchModeFilterIndex);
                        % Run segment button functionality
                        SegmentAndDisplayImage(app);
                        % Accept segmentation
                        label = bwlabel(~logical(app.bw_obj),4);
                        SaveMaskToMaskFile(app, label);
                    end
                end
                app.Prompt.FontColor = [0 0 0];
                app.Prompt.Text = 'Batch Mode - Initial Segmentation Completed.';
                app.FiberPredictionButton.Enable = 'on';
                app.SegmentationParameters.Visible = 'off';
            end
        end

        % Value changed function: SegmentationThresholdSlider
        function SegmentationThresholdSliderValueChanged(app, event)
            value = app.SegmentationThresholdSlider.Value;
            app.SegmentationThresholdSlider.Value = round(value);
            
        end

        % Button pushed function: YesButton
        function YesButtonPushed(app, event)
            uiresume(app.UIFigure);
        end

        % Button pushed function: NoButton
        function NoButtonPushed(app, event)
            app.notfiber = 1;
            uiresume(app.UIFigure);
        end

        % Button pushed function: ManualSortingButton
        function ManualSortingButtonPushed(app, event)
            app.FiberPredictionControlPanel.Visible = 'off';
            app.ManualSortingButton.Enable = 'off';
            app.ImageBackground.Visible = 'off';
            uiresume(app.UIFigure);
        end

        % Button pushed function: RemoveNonfibersButton
        function RemoveNonfibersButtonPushed(app, event)
            app.RemoveNonfibersButton.Enable = 'off';
            app.FinishManualFilteringButton.Enable = 'on';
            label = bwlabel(app.bw_obj,4);
            bw_pos = app.bw_obj;
            bw_all = app.bw_obj;
            base_flat_img = flattenMaskOverlay(app.orig_img,bw_all,0.1,'w');
            app.num_obj = max(max(label));
            app.done = 0;
            pon = true(app.num_obj,1); % logical ones array to determine if a region has been selected
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
           
            userStopped = false;
            
            while ~app.done && ~userStopped
                app.Prompt.Text = 'Click on regions for removal, click esc to finish manual filtering.';
                phand = drawpoint(app.UIAxes,'Color','w');
                
                if ~isvalid(phand) || isempty(phand.Position)
                    userStopped = true;
                end

                if ~app.done && ~userStopped
                    pos = round(phand.Position);
                    xp = pos(1);
                    yp = pos(2);
                    delete(phand)
                end

                if ~app.done && ~userStopped
                    % Continue if clicked point is out of bounds
                    if ~IsROIPositionInBound(app, xp, yp)
                        continue
                    end
                    regS = label(yp,xp);
                    if regS == 0
                        continue
                    end
                    pon(regS) = ~pon(regS);
                    idx = find(label == regS);
                    if pon(regS)
                        bw_pos(idx) = 1;
                    else
                        bw_pos(idx) = 0;
                    end
                    flat_img = flattenMaskOverlay(base_flat_img,bw_pos,0.5,'w');
                    imshow(flat_img,'Parent',app.UIAxes);
                end
            end
            app.bw_obj = bw_pos;
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            label = bwlabel(app.bw_obj,4);
            SaveMaskToMaskFile(app, label);
            app.ManualFilterControls.Visible = 'off';
            EnableMenuBarButtons(app);
            app.Prompt.Text = '';
            app.RemoveNonfibersButton.Enable = 'on';
        end

        % Button pushed function: FinishManualFilteringButton
        function FinishManualFilteringButtonPushed(app, event)
            app.Prompt.Text = 'Click anywhere on the screen or esc to continue';
            app.FinishManualFilteringButton.Enable = 'off';
            app.done = 1;
        end

        % Button pushed function: CalculateFiberProperties
        function CalculateFiberPropertiesPushed(app, event)
            app.PropertiesPanel.Visible = 'on';
            app.ImageBackground.Visible = 'off';

            app.WritetoExcelButton.Enable = 'on';
            app.pix_size = app.PixelSizeFiberProperties.Value;
            app.output_path = app.FiberPropertiesDataOutputFolder.Value;
            pix_area = app.pix_size^2;
            label = bwlabel(app.bw_obj,4);
            app.num_obj = max(max(label));
            
            % Determine fiber properties
            app.props = regionprops('table',label,'Area','MinFeretProperties','MaxFeretProperties','Centroid');
            app.props.Area = app.props.Area * pix_area ;
            app.props.MaxFeretDiameter = app.props.MaxFeretDiameter * app.pix_size;
            app.props.MinFeretDiameter = app.props.MinFeretDiameter * app.pix_size;
            
            % Plot properties
            histogram(app.FeretAxes,app.props.MinFeretDiameter,20);
            histogram(app.FiberSizeAxes,app.props.Area,20);
            
            
        end

        % Button pushed function: DoneFiberProperties
        function DoneFiberPropertiesPushed(app, event)
            app.Prompt.Text = '';
            app.FiberPropertiesControlPanel.Visible = 'off';
            app.PropertiesPanel.Visible = 'off';
            app.ImageBackground.Visible = 'on';
            EnableMenuBarButtons(app);
        end

        % Button pushed function: WritetoExcelButton
        function WritetoExcelButtonPushed(app, event)
            app.FiberPropertiesFileWriteStatusLabel.Text = 'Writing to Excel...';
            cents = cat(1,app.props.Centroid);
            rprop = [cents(:,2) cents(:,1) app.props.MaxFeretDiameter app.props.MaxFeretAngle app.props.MinFeretDiameter app.props.MinFeretAngle app.props.Area];
            propsum(1,:) = mean(rprop,1);
            propsum(2,:) = std(rprop,1);
            propsum(3,:) = propsum(2,:)./sqrt(app.num_obj);
            
            header{1,1} = 'Filename';
            header{1,2} = app.Files{4};
            header{2,1} = 'Number of fibers';
            header{2,2} = app.num_obj;
            
            header{3,2} = 'Centroid X';
            header{3,3} = 'Centroid Y';
            header{3,4} = 'Max Feret Diameter (um)';
            header{3,5} = 'Max Feret Angle';
            header{3,6} = 'Min Feret Diameter (um)';
            header{3,7} = 'Min Feret Angle';
            header{3,8} = 'Area (um^2)';
            
            sums{1,1} = 'Mean';
            sums{2,1} = 'Std Dev';
            sums{3,1} = 'SEM';
            
            sums = [sums num2cell(propsum)];
            
            fibs = 1:app.num_obj;
            data = [num2cell(fibs') num2cell(rprop)];
            
            out = [header; sums; data];
            
            % Create folder if directory does not exist for excel input
            CreateFolderIfDirectoryIsNonexistent(app, app.FiberPropertiesDataOutputFolder.Value);
            cd(app.FiberPropertiesDataOutputFolder.Value)

            writecell(out,[app.Files{4} '_Properties.xlsx'],'Range','A5');
            app.props = 0;
            cd(app.Files{3})
            app.FiberPropertiesFileWriteStatusLabel.Text = 'Write to Excel done!';
        end

        % Value changed function: SortingThresholdSlider
        function SortingThresholdSliderValueChanged(app, event)
            value = app.SortingThresholdSlider.Value;
            app.SortingThresholdSlider.Value = round(value,1);
        end

        % Button pushed function: CalculateCentralNuclei
        function CalculateCentralNucleiPushed(app, event)
            app.ImageBackground.Visible = 'off';
            app.CentralNucleiPanel.Visible = 'on';
            app.ThresholdCentralNuclei.Enable = 'off';
            app.AdjustCentralNuclei.Enable = 'off';
            app.AcceptCentralNuclei.Enable = 'off';

            app.Prompt.Text = '';
            app.CalculateCentralNuclei.Enable = 'off';
            app.DoneCentralNuclei.Enable = 'off';
            app.pix_size = app.PixelSizeCentralNuclei.Value;
            min_nuc_pix = app.MinimumNucleusSizeum2EditField.Value/(app.pix_size^2);
            border = app.DistancefromborderEditField.Value;
            border_pix = border/app.pix_size;
            label_org = bwconncomp(app.bw_obj,4);
            num_fib = label_org.NumObjects;
            
            % Create border region
            inv_img = imcomplement(app.bw_obj);
            dist = bwdist(inv_img);
            inv_brd = (dist > border_pix);
            
            % Define Nuclei
            nuc_obj = app.orig_img_multispectral(:,:,str2double(app.NucleiColorDropDown.Value));
            se = strel('disk',12);
            tophatFiltered = imtophat(nuc_obj,se);
            nuc_fil = imadjust(tophatFiltered);
            threshes = multithresh(nuc_fil,10);
            app.thresh_nuc = double(threshes(2))/255;
            
            app.ThresholdCentralNuclei.Enable = 'on';
            app.AdjustCentralNuclei.Enable = 'on';
            app.AcceptCentralNuclei.Enable = 'on';
            app.ThresholdCentralNuclei.Value = app.thresh_nuc*255;
            
            app.CentralNucleiAdj = 1;
            
            imshow(app.orig_img, 'Parent', app.CentralNucleiAxesL);
            while app.CentralNucleiAdj
                nuc_bw = imbinarize(nuc_fil,app.thresh_nuc);
                nuc_cen = nuc_bw.*inv_brd;
            
                nprop = regionprops(label_org,nuc_cen,'PixelValues');
                app.cen_pix = zeros(num_fib,1);
                    for i = 2:num_fib
                        app.cen_pix(i) = sum(nprop(i).PixelValues);
                    end
                app.cen_nuc = logical(app.cen_pix >= min_nuc_pix);

                % Label fibers with central nuclei
                app.fprop = regionprops(label_org,'PixelIdxList','Area');
                central_nuclei_img = app.bw_obj.*0.3;
                central_nuclei_img(vertcat(app.fprop(app.cen_nuc).PixelIdxList)) = 1;

                pos_img = flattenMaskOverlay(central_nuclei_img,nuc_bw,0.6,'b');
                imshow(pos_img,'Parent',app.CentralNucleiAxesR);
                linkaxes([app.CentralNucleiAxesL, app.CentralNucleiAxesR]);
                uiwait(app.UIFigure);
            end
            
        end

        % Button pushed function: CentralNucleiExcelWrite
        function CentralNucleiExcelWriteButtonPushed(app, event)
            app.CentralNucleiFileWriteStatusLabel.Text = 'Writing to Excel...';

            % Create folder if directory does not exist for excel input
            CreateFolderIfDirectoryIsNonexistent(app, app.CentralNucleiDataOutputFolder.Value);
            cd(app.CentralNucleiDataOutputFolder.Value)

            header{1,1} = 'Border (um)';
            header{1,2} = 'Minimum Nuclear Size (um^2)';
            header{1,3} = 'Nuclear Filter (arb)';
            
            header{2,1} = app.DistancefromborderEditField.Value;
            header{2,2} = app.MinimumNucleusSizeum2EditField.Value*(app.pix_size^2);
            header{2,3} = app.thresh_nuc;
            
            header{3,1} = 'Average Fiber Size';
            header{3,2} = 'Average Blue Center';
            header{3,3} = 'Percent Positive';
            
            header{4,1} = mean([app.fprop.Area]).*(app.pix_size^2);
            header{4,2} = mean(app.cen_pix).*(app.pix_size^2);
            header{4,3} = mean(app.cen_nuc);
            
            header{5,1} = 'Positive Fiber Size';
            header{5,2} = 'Positive Fiber Blue Center';
            header{5,3} = 'Number Positive';
            
            header{6,1} = mean([app.fprop(app.cen_nuc).Area]).*(app.pix_size^2);
            header{6,2} = mean(app.cen_pix(app.cen_nuc)).*(app.pix_size^2);
            header{6,3} = sum(app.cen_nuc);
            
            header{7,1} = 'Negative Fiber Size';
            header{7,2} = 'Negative Fiber Blue Center';
            header{7,3} = 'Number Negative';
            
            header{8,1} = mean([app.fprop(~app.cen_nuc).Area]).*(app.pix_size^2);
            header{8,2} = mean(app.cen_nuc(~app.cen_nuc)).*(app.pix_size^2);
            header{8,3} = sum(~app.cen_nuc);
            
            header{10,1} = 'Fiber Size';
            header{10,2} = 'Fiber Blue Center';
            header{10,3} = 'Fiber Positive CNF';
            
            farea = [app.fprop.Area]';
            farea = farea.*(app.pix_size^2);
            data = cat(2,farea,app.cen_pix.*app.pix_size^2,app.cen_nuc) ;
            out = cat(1,header,num2cell(data));
            writecell(out,[app.Files{4} '_Properties.xlsx'],'Range','I1');
            app.CentralNucleiFileWriteStatusLabel.Text = 'Write to Excel done!';
            cd(app.Files{3})
            app.fprop = 0;
            app.cen_nuc = 0;
            app.cen_pix = 0;
        end

        % Button pushed function: DoneCentralNuclei
        function DoneCentralNucleiPushed(app, event)
            app.Prompt.Text = '';
            app.CentralNucleiControlPanel.Visible = 'off';
            app.CentralNucleiPanel.Visible = 'off';
            app.ImageBackground.Visible = 'on';
            EnableMenuBarButtons(app);
        end

        % Button pushed function: CalculateFiberTyping
        function CalculateFiberTypingButtonPushed(app, event)
            app.ImageBackground.Visible = 'off';
            app.FiberTypingPanel.Visible = 'on';

            app.CalculateFiberTyping.Enable = 'off';
            app.PixelSizeFiberTyping.Enable = 'off';
            app.FiberTypeColorDropDown.Enable = 'off';
            app.DoneFiberTyping.Enable = 'off';
            app.WritetoExcelFT.Enable = 'off';
            app.pix_size = app.PixelSizeFiberTyping.Value;
            pix_area = app.pix_size^2;
            
            % Create Labelled Fibers
            label = bwlabel(app.bw_obj,4);
            app.num_obj = max(max(label));
            
            % Threshold Fiber Types
            fti = app.orig_img_multispectral(:,:,str2double(app.FiberTypeColorDropDown.Value));
            threshes = multithresh(fti,10);
            app.cutoff_avg = threshes(2);
            
            % Fiber Properties
            rprop = regionprops(bwconncomp(app.bw_obj,4),fti,'MeanIntensity','Centroid','Area','PixelIdxList');
            app.ave_g = [rprop.MeanIntensity];
            app.areas = [rprop.Area];
            
            app.ThresholdEditField.Enable = 'on';
            app.ThresholdEditFieldLabel.Enable = 'on';
            app.ThresholdEditField.Value = double(app.cutoff_avg);
            app.AdjustButton.Enable = 'on';
            app.AcceptButton.Enable = 'on';
            
            app.FT_Adj = 1;
            while app.FT_Adj
                app.ponf = false(app.num_obj,1);
                app.ponf(app.ave_g > app.cutoff_avg) = 1;
                img_out = single(app.bw_obj) .* 0.3;
                p_ind = find(app.ponf);
                for i = 1:length(p_ind)
                    img_out(label == p_ind(i)) = 1;
                end
                
                x = [app.cutoff_avg, app.cutoff_avg];
                y = [0,25];
                
                [tot_n, cen] = histcounts((app.areas .* pix_area),20);
                [pos_n, cen] = histcounts((app.areas(app.ponf) .* pix_area),cen);
                
                imshow(app.orig_img,'Parent',app.FiberTypingAxesL);
                imshow(img_out,'Parent',app.FiberTypingAxesR);
                linkaxes([app.FiberTypingAxesL, app.FiberTypingAxesR]);

                histogram(app.ave_g,255,'Parent',app.FThistL);
                hold(app.FThistL,'on');
                line(app.FThistL,x,y,'LineWidth',2,'Color','r');
                hold(app.FThistL,'off');
                histogram('BinEdges',cen,'BinCounts',tot_n,'Parent',app.FThistR);
                hold(app.FThistR,'on');
                histogram('BinEdges',cen,'BinCounts',pos_n,'FaceColor','r','Parent',app.FThistR);
                hold(app.FThistR,'off');
                uiwait(app.UIFigure);
                
            end
        end

        % Button pushed function: AdjustButton
        function AdjustButtonPushed(app, event)
            app.cutoff_avg = app.ThresholdEditField.Value;
            uiresume(app.UIFigure);
        end

        % Button pushed function: AcceptButton
        function AcceptButtonPushed(app, event)
            app.FT_Adj = 0;
            app.ThresholdEditField.Enable = 'off';
            app.AdjustButton.Enable = 'off';
            app.AcceptButton.Enable = 'off';
            app.CalculateFiberTyping.Enable = 'on';
            app.WritetoExcelFT.Enable = 'on';
            app.DoneFiberTyping.Enable = 'on';
            uiresume(app.UIFigure);
        end

        % Button pushed function: WritetoExcelFT
        function WritetoExcelFTButtonPushed(app, event)
            app.FiberTypingFileWriteStatusLabel.Text = 'Writing to Excel...';

            % Create folder if directory does not exist for excel input
            CreateFolderIfDirectoryIsNonexistent(app, app.FiberTypingDataOutputFolder.Value);
            cd(app.FiberTypingDataOutputFolder.Value)

            pix_area = app.pix_size^2;
            header{1,1} = 'Average Fiber Size';
            header{1,2} = 'Average Fiber Intensity';
            header{1,3} = 'Percent Positibe';
            
            header{2,1} = mean(app.areas).*pix_area;
            header{2,2} = mean(app.ave_g);
            header{2,3} = mean(app.ponf);
            
            header{3,1} = 'Positive Fiber Size';
            header{3,2} = 'Positive Fiber Intensity';
            header{3,3} = 'Number Positive';
            
            header{4,1} = mean(app.areas(app.ponf)).*pix_area;
            header{4,2} = mean(app.ave_g(app.ponf));
            header{4,3} = sum(app.ponf);
            
            header{5,1} = 'Negative Fiber Size';
            header{5,2} = 'Negative Fiber Intensity';
            header{5,3} = 'Number Negative';
            
            header{6,1} = mean(app.areas(~app.ponf)).*pix_area;
            header{6,2} = mean(app.ave_g(~app.ponf));
            header{6,3} = app.num_obj - sum(app.ponf);
            
            header{10,1} = 'Fiber Size';
            header{10,2} = 'Fiber Intensity';
            header{10,3} = 'Fiber Positive';
            
            out_data = zeros(app.num_obj,3);
            out_data(:,1) = app.areas.*pix_area;
            out_data(:,2) = app.ave_g;
            out_data(:,3) = app.ponf;
            
            out_file = cat(1,header,num2cell(out_data));
            if str2double(app.FiberTypeColorDropDown.Value) == 2
                writecell(out_file,[app.Files{4} '_Properties.xlsx'],'Range','L1');
            elseif str2double(app.FiberTypeColorDropDown.Value) == 1
                writecell(out_file,[app.Files{4} '_Properties.xlsx'],'Range','O1');
            elseif str2double(app.FiberTypeColorDropDown.Value) == 3
                writecell(out_file,[app.Files{4} '_Properties.xlsx'],'Range','R1');
            end
            app.FiberTypingFileWriteStatusLabel.Text = 'Write to Excel done!';

            cd(app.Files{3})
            app.ponf = 0;
            app.ave_g = 0;
            app.areas = 0;
        end

        % Button pushed function: DoneFiberTyping
        function DoneFiberTypingButtonPushed(app, event)
            app.Prompt.Text = '';
            app.ImageBackground.Visible = 'on';
            app.FiberTypingControlPanel.Visible = 'off';
            app.FiberTypingPanel.Visible = 'off';
            EnableMenuBarButtons(app);
        end

        % Button pushed function: AdjustCentralNuclei
        function AdjustCentralNucleiButtonPushed(app, event)
            app.thresh_nuc = app.ThresholdCentralNuclei.Value/255;
            uiresume(app.UIFigure);
        end

        % Button pushed function: AcceptCentralNuclei
        function AcceptCentralNucleiButtonPushed(app, event)
            app.CentralNucleiAdj = 0;
            app.ThresholdCentralNuclei.Enable = 'off';
            app.AdjustCentralNuclei.Enable = 'off';
            app.AcceptCentralNuclei.Enable = 'off';
            app.CalculateCentralNuclei.Enable = 'on';
            app.CentralNucleiExcelWrite.Enable = 'on';
            app.DoneCentralNuclei.Enable = 'on';
            uiresume(app.UIFigure);
        end

        % Button pushed function: CalculateNonfiberObjects
        function CalculateNonfiberObjectsButtonPushed(app, event)
            app.ImageBackground.Visible = 'off';
            app.NonfiberPanel.Visible = 'on';
            app.NonfiberThreshold.Enable = 'off';
            app.NonfiberAdjust.Enable = 'off';
            app.NonfiberAccept.Enable = 'off';

            img_org = app.orig_img_multispectral;
            app.CalculateNonfiberObjects.Enable = 'off';
            app.DoneNonfiber.Enable = 'off';
            app.pix_size = app.PixelSizeNonfiberObjects.Value;
            ch_obj = img_org(:,:,str2double(app.NonfiberObjectsColorDropDown.Value));
            % smoothed image
            se = strel('disk',12);
            tophatFiltered = imtophat(ch_obj, se);
            ch_fil = imadjust(tophatFiltered);
            % determine threshold value
            threshes = multithresh(ch_fil,10);
            app.thresh_nf = double(threshes(1))/255;
            
            app.NonfiberAccept.Enable = 'on';
            app.NonfiberAdjust.Enable = 'on';
            app.NonfiberThreshold.Enable = 'on';
            app.NonfiberThreshold.Value = app.thresh_nf*255;
            
            app.Obj_Adj = 1;
            
            imshow(app.orig_img, 'Parent', app.NonfiberAxesL);
            while app.Obj_Adj
                ch_bw = imbinarize(ch_fil,app.thresh_nf);
                imshow(ch_bw,'Parent',app.NonfiberAxesR);
                linkaxes([app.NonfiberAxesL, app.NonfiberAxesR]);
                
                app.nf_bw_obj = ch_bw;
                uiwait(app.UIFigure);
                
                if app.Obj_Adj
                    app.thresh_nf = app.NonfiberThreshold.Value/255;
                end
                
            end

            nfprops = regionprops(bwconncomp(ch_bw,4),'Centroid','Area');
            area_nf = [nfprops.Area]'*app.pix_size^2;
            cents_nf = cat(1,nfprops.Centroid);
            app.num_nf = length(area_nf);
            app.nf_data = [area_nf cents_nf];
                
        end

        % Button pushed function: NonfiberAdjust
        function NonfiberAdjustButtonPushed(app, event)
            app.Obj_Adj = 1;
            uiresume(app.UIFigure);
        end

        % Button pushed function: NonfiberAccept
        function NonfiberAcceptButtonPushed(app, event)
            app.Obj_Adj = 0;
            app.NonfiberThreshold.Enable = 'off';
            app.NonfiberAccept.Enable = 'off';
            app.NonfiberAdjust.Enable = 'off';
            app.CalculateNonfiberObjects.Enable = 'on';
            app.WritetoExcelNonfiber.Enable = 'on';
            app.DoneNonfiber.Enable = 'on';
            uiresume(app.UIFigure);
        end

        % Button pushed function: WritetoExcelNonfiber
        function WritetoExcelNonfiberButtonPushed(app, event)
            app.NonfiberObjectsFileWriteStatusLabel.Text = 'Writing to Excel...';

            % Create folder if directory does not exist for excel input
            CreateFolderIfDirectoryIsNonexistent(app, app.NonfiberObjectsDataOutputFolder.Value);
            cd(app.NonfiberObjectsDataOutputFolder.Value)
            
            header{1,1} = 'Threshold';
            header{2,1} = 'Number of Objects';
            header{1,2} = app.thresh_nf*255;
            header{2,2} = app.num_nf;
            
            header{4,1} = 'Area (um2)';
            header{4,2} = 'X centroid';
            header{4,3} = 'Y centroid';
            
            out = cat(1,header,num2cell(app.nf_data));
            
            writecell(out, [app.Files{4} '_Properties.xlsx'], 'Range','P7');
            app.NonfiberObjectsFileWriteStatusLabel.Text = 'Write to Excel done!';
            cd(app.Files{3})
            
        end

        % Button pushed function: DoneNonfiber
        function DoneNonfiberButtonPushed(app, event)
            app.ImageBackground.Visible = 'on';
            app.thresh_nf = 0;
            app.nf_data = 0;
            app.Prompt.Text = '';
            app.NonfiberPanel.Visible = 'off';
            app.NonfiberObjectsControlPanel.Visible = 'off';
            SaveNFMaskToMaskFile(app, app.nf_bw_obj);
            EnableMenuBarButtons(app);
        end

        % Button pushed function: InitialSegmentationButton
        function InitialSegmentationButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.ImageBackground.Visible = 'on';
            app.AcceptSegmentationButton.Enable = 'off';

            % Display the image
            imshow(app.orig_img,'Parent',app.UIAxes);

            app.SegmentationParameters.Visible = 'on';
            app.FiberOutlineChannelColorBox.Visible = 'on';

            if app.IsBatchMode == 1
                app.ImageBackground.Visible = 'on';
                app.Prompt.Text = '';
                app.FiberPredictionControlPanel.Visible = 'off';
            end
        end

        % Button pushed function: ManualSegmentationButton
        function ManualSegmentationButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.ImageBackground.Visible = 'on';
            app.ManualSegmentationControls.Visible = 'on';
            app.bw_obj = ReadMaskFromMaskFile(app);
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,1,'w'),'Parent',app.UIAxes);
        end

        % Button pushed function: FiberPredictionButton
        function FiberPredictionButtonPushed(app, event)
            if app.IsBatchMode == 0
                DisableMenuBarButtonsAndClearFileLabels(app);
                app.ImageBackground.Visible = 'on';
                app.FiberPredictionControlPanel.Visible = 'on';
                % acquire mask and show over image
                app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
                app.bw_obj = imclearborder(app.bw_obj,4);
                imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
                app.FilterButton.Enable = 'on';
                app.SortingThresholdSlider.Enable = 'on';
                app.pix_size = app.PixelSizeFiberProperties.Value;
            else
                app.Prompt.Text = '';
                app.ImageBackground.Visible = 'on';
                app.InitialSegmentationButton.Enable = 'off';
                app.FiberPredictionControlPanel.Visible = 'on';
                app.ManualSortingButton.Enable = 'off';
                app.ManualSortingButton.Visible = 'on';
                app.SortingThresholdSlider.Visible = 'on';
                app.SortingThresholdSlider.Enable = 'off';
                app.FilterButton.Enable = 'on';
                app.pix_size = app.PixelSizeFiberProperties.Value;
            end
        end

        % Button pushed function: ManualFiberFilterButton
        function ManualFiberFilterButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.ImageBackground.Visible = 'on';
            app.ManualFilterControls.Visible = 'on';
            app.RemoveNonfibersButton.Enable = 'on';
            app.FinishManualFilteringButton.Enable = 'off';
            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
        end

        % Button pushed function: FiberPropertiesButton
        function FiberPropertiesButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.FiberPropertiesControlPanel.Visible = 'on';
            app.FiberPropertiesFileWriteStatusLabel.Text = '';
            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
            app.bw_obj = imclearborder(app.bw_obj,4);
        end

        % Button pushed function: CentralNucleiButton
        function CentralNucleiButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);

            app.CentralNucleiControlPanel.Visible = 'on';
            app.CentralNucleiExcelWrite.Enable = 'off';
            app.CentralNucleiChannelColorBox.Visible = 'on';
            app.CentralNucleiFileWriteStatusLabel.Text = '';

            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
        end

        % Button pushed function: FiberTypingButton
        function FiberTypingButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);

            app.FiberTypingFileWriteStatusLabel.Text = '';
            app.FiberTypingControlPanel.Visible = 'on';
            app.PixelSizeFiberTyping.Enable = 'on';
            app.FiberTypeColorDropDown.Enable = 'on';
            app.WritetoExcelFT.Enable = 'off';
            app.FiberTypingChannelColorBox.Visible = 'on';

            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));
        end

        % Button pushed function: NonfiberObjectsButton
        function NonfiberObjectsButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.NonfiberObjectsFileWriteStatusLabel.Text = '';
            app.NonfiberObjectsControlPanel.Visible = 'on';
            app.WritetoExcelNonfiber.Enable = 'off';
            app.NonfiberChannelColorBox.Visible = 'on';
        end

        % Button pushed function: DetectValueButton
        function DetectValueButtonPushed(app, event)
            foc = app.FiberOutlineColorDropDown.Value;
            foc = str2double(foc);
            lam = app.orig_img_multispectral(:,:,foc);  % fiber outline color
            hist = imhist(lam);
            hist = hist(1:end-1);
            AverageIntensity = mean2(lam);
            x = linspace(0,254,255);
            x = x';
            y = fit(x,hist,'Gauss1');
            %peak = y.a1;
            PeakofGaussian = round(y.b1);
            stddev = round(y.c1);
            hist_adj = hist;
            hist_adj(1:(2*stddev+PeakofGaussian)) = 0;
            AveragePositiveIntensity = sum(x.*hist_adj)/sum(hist_adj);
            predictors = table(AverageIntensity, PeakofGaussian, AveragePositiveIntensity);
            classifier = app.segmodel.segModel.RegressionSVM;
            value = predict(classifier,predictors);
            app.SegmentationThresholdSlider.Value = round(value);
        end

        % Button pushed function: FinishDrawingButton
        function FinishDrawingButtonPushed(app, event)
            app.done = 1;
            uiresume(app.UIFigure);
            app.StartDrawingButton.Enable = 'on';
            app.AcceptLineButton.Enable = 'off';
            app.StartMergingButton.Enable = 'on';
            app.FinishDrawingButton.Enable = 'off';
            app.CloseManualSegmentationButton.Enable = 'on';

            imshow(flattenMaskOverlay(app.orig_img, app.bw_obj, 1, 'w'), 'Parent', app.UIAxes);

            label = bwlabel(~logical(app.bw_obj),4);
            SaveMaskToMaskFile(app, label);

            app.Prompt.Text = '';
        end

        % Button pushed function: StartMergingButton
        function StartMergingButtonPushed(app, event)
            app.StartDrawingButton.Enable = 'off';
            app.StartMergingButton.Enable = 'off';
            app.CloseManualSegmentationButton.Enable = 'off';

            % Load mask
            app.bw_obj = imcomplement(ReadMaskFromMaskFile(app));

            % Initialize constant
            NONE_REGION_SELECTED = -1; % Dummy variable for first_region_to_merge when no region is selected.

            % Initialize local variables
            object_labels = bwlabel(app.bw_obj,4);
            bw_selected = zeros(size(app.bw_obj), "logical");
            first_region_to_merge = NONE_REGION_SELECTED;
            is_there_an_merge_error_to_report_to_user = 0;

            % Display objects.
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            app.done = 0;
            while ~app.done
                if first_region_to_merge == NONE_REGION_SELECTED
                    app.Prompt.Text = 'Select first region to merge or press ESC to finish merge mode';
                else
                    if is_there_an_merge_error_to_report_to_user
                        app.Prompt.Text = 'Merge unsuccessful, please click adjacent region to merge or click selected region to unselect or press ECS to finish merge mode';
                        is_there_an_merge_error_to_report_to_user = 0;
                    else
                        app.Prompt.Text = 'Click adjacent region to merge or click selected region to unselect or press ECS to finish merge mode';
                    end
                end

                roi = drawpoint(app.UIAxes,'Color','w');

                if ~isvalid(roi) || isempty(roi.Position)
                    app.done = 1;
                end

                if ~app.done
                    pos = round(roi.Position);
                    xp = pos(1);
                    yp = pos(2);
                    delete(roi)

                    % Continue if clicked point is out of bounds
                    if ~IsROIPositionInBound(app, xp, yp)
                        continue
                    end

                    % Continue if clicked point is not a region
                    clicked_region = object_labels(yp,xp);
                    if clicked_region == 0
                        continue
                    end

                    is_user_selecting_first_of_two_objects_to_merge = first_region_to_merge == NONE_REGION_SELECTED;
                    is_user_unselecting_region = first_region_to_merge == clicked_region;
                    if is_user_selecting_first_of_two_objects_to_merge
                        bw_selected(object_labels == clicked_region) = 1;
                        first_region_to_merge = clicked_region;
                    elseif is_user_unselecting_region
                        bw_selected(object_labels == clicked_region) = 0;
                        first_region_to_merge = NONE_REGION_SELECTED;
                    else
                        %%%% Merging
                        app.Prompt.Text = 'Merging regions...';
                        second_region_to_merge = clicked_region;
                        [app.bw_obj, is_merged_successful] = MergeObjects(app, object_labels, first_region_to_merge, second_region_to_merge);

                        if is_merged_successful
                            %%%% Reset Local Variables

                            % Two connected components have been merged, so we need to relabel
                            object_labels = bwlabel(app.bw_obj, 4);

                            % Select the newly created object
                            new_object_label = object_labels(yp,xp);
                            bw_selected = object_labels == new_object_label;
                            first_region_to_merge = new_object_label;
                        else
                            is_there_an_merge_error_to_report_to_user = 1;
                        end
                    end

                    % Draw objects and selection
                    flat_img = flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w');
                    flat_img = flattenMaskOverlay(flat_img,bw_selected,0.8,'w');
                    imshow(flat_img,'Parent',app.UIAxes);
                end
            end
            object_labels = bwlabel(app.bw_obj,4);
            app.num_obj = max(max(object_labels));
            SaveMaskToMaskFile(app, object_labels);

            % Go back to drawing lines instead of regions
            app.bw_obj = imcomplement(app.bw_obj);
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,1,'w'),'Parent',app.UIAxes);

            app.StartDrawingButton.Enable = 'on';
            app.StartMergingButton.Enable = 'on';
            app.CloseManualSegmentationButton.Enable = 'on';
            app.Prompt.Text = '';
        end

        % Value changed function: NonfiberObjectsColorDropDown
        function NonfiberObjectsColorDropDownValueChanged(app, event)
            % Retreive the index of the color channel
            color_channel_index = str2double(app.NonfiberObjectsColorDropDown.Value);

            % Get RGB values of color channel and convert to numeric matrix
            RGBValueOfColorChannel = app.channelRGB(color_channel_index, :);
            RGB1DValue = RGBValueOfColorChannel;

            % Update color on color box
            app.NonfiberChannelColorBox.Color = RGB1DValue;
            
        end

        % Value changed function: FiberTypeColorDropDown
        function FiberTypeColorValueChanged(app, event)
            % Retreive the index of the color channel
            color_channel_index = str2double(app.FiberTypeColorDropDown.Value);

            % Get RGB values of color channel and convert to numeric matrix
            RGBValueOfColorChannel = app.channelRGB(color_channel_index, :);
            RGB1DValue = RGBValueOfColorChannel;

            % Update color on color box
            app.FiberTypingChannelColorBox.Color = RGB1DValue;
            
        end

        % Value changed function: NucleiColorDropDown
        function NucleiColorValueChanged(app, event)
            % Retreive the index of the color channel
            color_channel_index = str2double(app.NucleiColorDropDown.Value);

            % Get RGB values of color channel and convert to numeric matrix
            RGBValueOfColorChannel = app.channelRGB(color_channel_index, :);
            RGB1DValue = RGBValueOfColorChannel;

            % Update color on color box
            app.CentralNucleiChannelColorBox.Color = RGB1DValue;
            
        end

        % Value changed function: FiberOutlineColorDropDown
        function FiberOutlineColorValueChanged(app, event)
            % Retreive the index of the color channel
            color_channel_index = str2double(app.FiberOutlineColorDropDown.Value);

            % Get RGB values of color channel and convert to numeric matrix
            RGBValueOfColorChannel = app.channelRGB(color_channel_index, :);
            RGB1DValue = RGBValueOfColorChannel;

            % Update color on color box
            app.FiberOutlineChannelColorBox.Color = RGB1DValue;
            
        end

        % Value changed function: PixelSizeNonfiberObjects
        function PixelSizeNonfiberObjectsValueChanged(app, event)
            app.pix_size = app.PixelSizeNonfiberObjects.Value;
            SyncPixelSize(app);
        end

        % Value changed function: PixelSizeFiberTyping
        function PixelSizeFiberTypingValueChanged(app, event)
            app.pix_size = app.PixelSizeFiberTyping.Value;
            SyncPixelSize(app);
        end

        % Value changed function: PixelSizeCentralNuclei
        function PixelSizeCentralNucleiValueChanged(app, event)
            app.pix_size = app.PixelSizeCentralNuclei.Value;
            SyncPixelSize(app);
        end

        % Value changed function: PixelSizeFiberProperties
        function PixelSizeFiberPropertiesValueChanged(app, event)
            app.pix_size = app.PixelSizeFiberProperties.Value;
            SyncPixelSize(app);
        end

        % Button pushed function: NonfiberClassificationButton
        function NonfiberClassificationButtonPushed(app, event)
            DisableMenuBarButtonsAndClearFileLabels(app);
            app.NonfiberClassificationFileWriteStatusLabel.Text = '';
            app.NonfiberClassificationControlPanel.Visible = 'on';
            app.PixelSizeNonfiberClassification.Enable = 'on';
            app.NonfiberClassificationColorDropDown.Enable = 'on';
            app.WritetoExcelNonfiberClassification.Enable = 'off';
            app.NonfiberClassificationChannelColorBox.Visible = 'on';
            app.PercentPositiveTextArea.Visible = 'on';
            app.nf_bw_obj = ReadNFMaskFromMaskFile(app);
            app.nf_mask = bwlabel(app.nf_bw_obj,4);
        end

        % Button pushed function: ClassifyNonfiberObjects
        function ClassifyNonfiberObjectsButtonPushed(app, event)
            app.ImageBackground.Visible = 'off';
            app.NonfiberClassificationPanel.Visible = 'on';

            % Labelled objects
            app.classified_nonfiber_num_obj = max(max(app.nf_mask));

            % Threshold Objects
            channel_name = app.NonfiberClassificationColorDropDown.Value;
            fti = app.orig_img_multispectral(:,:,str2double(channel_name));
            threshes = multithresh(fti,10);
            app.nonfiber_classification_cutoff_avg = threshes(2);
            app.NonfiberClassificationThreshold.Value = double(app.nonfiber_classification_cutoff_avg);

            % Display image
            imshow(app.orig_img,'Parent',app.NonfiberClassificationAxes_L);

            app.Obj_Adj = 1;
            while app.Obj_Adj
                % Fiber Properties
                rprop = regionprops(app.nf_mask,fti,'MeanIntensity','Centroid','Area','PixelIdxList');
                app.classified_nonfiber_ave_g = [rprop.MeanIntensity];
                app.classified_nonfiber_areas = [rprop.Area];

                % Determine which regions are above threshold
                app.classified_nonfiber_ponf = false(app.classified_nonfiber_num_obj,1); % logical zeros array of size num
                app.classified_nonfiber_ponf(app.classified_nonfiber_ave_g > app.nonfiber_classification_cutoff_avg) = 1; % set all elements where the intensity exceeds the threshold
                img_out = single(app.nf_bw_obj).* 0.3;

                p_ind = find(app.classified_nonfiber_ponf); % vector of qualifying regions
                app.classified_nonfiber_percentage = mean(app.classified_nonfiber_ponf) * 100;
                app.PercentPositiveTextArea.Value = string(app.classified_nonfiber_percentage) + " %";
                for i = 1:length(p_ind)
                    img_out(app.nf_mask == p_ind(i)) = 1; % whiten region
                end

                % Display image
                imshow(img_out,'Parent',app.NonfiberClassificationAxes_R);
                linkaxes([app.NonfiberClassificationAxes_L, app.NonfiberClassificationAxes_R]);

                app.NonfiberClassificationThreshold.Enable = 'on';
                app.NonfiberClassificationThreshold.Editable = 'on';
                app.NonfiberClassificationAccept.Enable = 'on';
                app.NonfiberClassificationAdjust.Enable = 'on';
                app.ClassifyNonfiberObjects.Enable = 'off';
                app.DoneNonfiberClassification.Enable = 'off';

                uiwait(app.UIFigure);

            end
        end

        % Button pushed function: NonfiberClassificationAdjust
        function NonfiberClassificationAdjustButtonPushed(app, event)
            app.nonfiber_classification_cutoff_avg = app.NonfiberClassificationThreshold.Value;         
            uiresume(app.UIFigure);
        end

        % Button pushed function: NonfiberClassificationAccept
        function NonfiberClassificationAcceptButtonPushed(app, event)
            app.Obj_Adj = 0;
            
            app.NonfiberClassificationThreshold.Enable = 'off';
            app.NonfiberClassificationAccept.Enable = 'off';
            app.NonfiberClassificationAdjust.Enable = 'off';
            app.ClassifyNonfiberObjects.Enable = 'on';
            app.WritetoExcelNonfiberClassification.Enable = 'on';
            app.DoneNonfiberClassification.Enable = 'on';

            uiresume(app.UIFigure);
        end

        % Button pushed function: WritetoExcelNonfiberClassification
        function WritetoExcelNonfiberClassificationButtonPushed(app, event)
            app.NonfiberClassificationFileWriteStatusLabel.Text = 'Writing to Excel...';

            % Create folder if directory does not exist for excel input
            CreateFolderIfDirectoryIsNonexistent(app, app.NonfiberClassificationDataOutputFolder.Value);
            cd(app.NonfiberClassificationDataOutputFolder.Value)

            pix_area = app.pix_size^2;
            header{1,1} = 'Average Classified Nonfiber Object Size';
            header{1,2} = 'Average Classified Nonfiber Object Intensity';
            header{1,3} = 'Percent Positive';

            header{2,1} = mean(app.classified_nonfiber_areas).*pix_area;
            header{2,2} = mean(app.classified_nonfiber_ave_g);
            header{2,3} = app.classified_nonfiber_percentage;

            header{3,1} = 'Positive Classified Nonfiber Object Size';
            header{3,2} = 'Positive Classified Nonfiber Object Intensity';
            header{3,3} = 'Number Positive';

            header{4,1} = mean(app.classified_nonfiber_areas(app.classified_nonfiber_ponf)).*pix_area;
            header{4,2} = mean(app.classified_nonfiber_ave_g(app.classified_nonfiber_ponf));
            header{4,3} = sum(app.classified_nonfiber_ponf);

            header{5,1} = 'Negative Classified Nonfiber Object Size';
            header{5,2} = 'Negative Classified Nonfiber Object Intensity';
            header{5,3} = 'Number Negative';

            header{6,1} = mean(app.classified_nonfiber_areas(~app.classified_nonfiber_ponf)).*pix_area;
            header{6,2} = mean(app.classified_nonfiber_ave_g(~app.classified_nonfiber_ponf));
            header{6,3} = app.classified_nonfiber_num_obj - sum(app.classified_nonfiber_ponf);

            header{10,1} = 'Classified Nonfiber Object Size';
            header{10,2} = 'Classified Nonfiber Object Intensity';
            header{10,3} = 'Classified Nonfiber Object Positive';

            out_data = zeros(app.classified_nonfiber_num_obj,3);
            out_data(:,1) = app.classified_nonfiber_areas.*pix_area;
            out_data(:,2) = app.classified_nonfiber_ave_g;
            out_data(:,3) = app.classified_nonfiber_ponf;

            out_file = cat(1,header,num2cell(out_data));
            
            writecell(out_file, [app.Files{4} '_Properties.xlsx'], 'Range','S1');
            app.NonfiberClassificationFileWriteStatusLabel.Text = 'Write to Excel done!';
            cd(app.Files{3})
            app.classified_nonfiber_ponf = 0;
            app.classified_nonfiber_ave_g = 0;
            app.classified_nonfiber_areas = 0;
        end

        % Button pushed function: DoneNonfiberClassification
        function DoneNonfiberClassificationButtonPushed(app, event)
            app.ImageBackground.Visible = 'on';
            app.nf_mask = 0;
            app.nf_bw_obj = 0;
            app.Prompt.Text = '';
            app.NonfiberPanel.Visible = 'off';
            app.NonfiberClassificationControlPanel.Visible = 'off';
            app.NonfiberClassificationPanel.Visible = 'off';
            EnableMenuBarButtons(app);
        end

        % Value changed function: PixelSizeNonfiberClassification
        function PixelSizeNonfiberClassificationValueChanged(app, event)
            app.pix_size = app.PixelSizeNonfiberClassification.Value;
            SyncPixelSize(app);
        end

        % Value changed function: PixelSizeFiberPrediction
        function PixelSizeFiberPredictionValueChanged(app, event)
            app.pix_size = app.PixelSizeFiberPrediction.Value;
            SyncPixelSize(app);
        end

        % Button pushed function: CloseInitialSegmentationButton
        function CloseInitialSegmentationButtonPushed(app, event)
            if app.IsBatchMode == 0
                app.SegmentationParameters.Visible = 'off';
                EnableMenuBarButtons(app);
            else
                app.FiberPredictionButton.Enable = 'on';
                app.InitialSegmentationButton.Enable = 'on';
                app.SegmentationParameters.Visible = 'off';
            end
        end

        % Value changed function: NonfiberClassificationColorDropDown
        function NonfiberClassificationColorDropDownValueChanged(app, event)
            % Retreive the index of the color channel
            color_channel_index = str2double(app.NonfiberClassificationColorDropDown.Value);

            % Get RGB values of color channel and convert to numeric matrix
            RGBValueOfColorChannel = app.channelRGB(color_channel_index, :);
            RGB1DValue = RGBValueOfColorChannel;

            % Update color on color box
            app.NonfiberClassificationChannelColorBox.Color = RGB1DValue;
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [1 1 1];
            app.UIFigure.Position = [100 100 1199 779];
            app.UIFigure.Name = 'MATLAB App';

            % Create SelectFileErrorLabel
            app.SelectFileErrorLabel = uilabel(app.UIFigure);
            app.SelectFileErrorLabel.VerticalAlignment = 'top';
            app.SelectFileErrorLabel.WordWrap = 'on';
            app.SelectFileErrorLabel.FontName = 'Avenir';
            app.SelectFileErrorLabel.FontColor = [1 0 0];
            app.SelectFileErrorLabel.Visible = 'off';
            app.SelectFileErrorLabel.Position = [40 536 234 49];
            app.SelectFileErrorLabel.Text = {'Some Error Message Will Go Here When There Is An Error'; ''; ''};

            % Create SelectFilesButton
            app.SelectFilesButton = uibutton(app.UIFigure, 'push');
            app.SelectFilesButton.ButtonPushedFcn = createCallbackFcn(app, @SelectFilesButtonPushed, true);
            app.SelectFilesButton.BackgroundColor = [1 1 1];
            app.SelectFilesButton.FontName = 'Avenir';
            app.SelectFilesButton.Position = [36 590 109 32];
            app.SelectFilesButton.Text = 'Select File(s)';

            % Create FilenameLabel
            app.FilenameLabel = uilabel(app.UIFigure);
            app.FilenameLabel.FontName = 'Avenir';
            app.FilenameLabel.Position = [154 596 130 22];
            app.FilenameLabel.Text = 'Filename';

            % Create Prompt
            app.Prompt = uilabel(app.UIFigure);
            app.Prompt.HorizontalAlignment = 'center';
            app.Prompt.FontName = 'Avenir';
            app.Prompt.Position = [289 650 824 22];
            app.Prompt.Text = '';

            % Create SortingAxesPanel
            app.SortingAxesPanel = uipanel(app.UIFigure);
            app.SortingAxesPanel.Title = 'Panel4';
            app.SortingAxesPanel.Visible = 'off';
            app.SortingAxesPanel.FontName = 'Avenir';
            app.SortingAxesPanel.Position = [285 28 890 633];

            % Create SortingAxesL
            app.SortingAxesL = uiaxes(app.SortingAxesPanel);
            xlabel(app.SortingAxesL, 'X')
            ylabel(app.SortingAxesL, 'Y')
            app.SortingAxesL.PlotBoxAspectRatio = [1 1.04306220095694 1];
            app.SortingAxesL.FontName = 'Avenir';
            app.SortingAxesL.XColor = 'none';
            app.SortingAxesL.YColor = 'none';
            app.SortingAxesL.Position = [12 140 412 438];

            % Create SortingAxesR
            app.SortingAxesR = uiaxes(app.SortingAxesPanel);
            xlabel(app.SortingAxesR, 'X')
            ylabel(app.SortingAxesR, 'Y')
            app.SortingAxesR.PlotBoxAspectRatio = [1 1.12082262210797 1];
            app.SortingAxesR.FontName = 'Avenir';
            app.SortingAxesR.XColor = 'none';
            app.SortingAxesR.YColor = 'none';
            app.SortingAxesR.Position = [467 140 412 438];

            % Create YesButton
            app.YesButton = uibutton(app.SortingAxesPanel, 'push');
            app.YesButton.ButtonPushedFcn = createCallbackFcn(app, @YesButtonPushed, true);
            app.YesButton.FontName = 'Avenir';
            app.YesButton.Position = [178 58 100 24];
            app.YesButton.Text = 'Yes';

            % Create NoButton
            app.NoButton = uibutton(app.SortingAxesPanel, 'push');
            app.NoButton.ButtonPushedFcn = createCallbackFcn(app, @NoButtonPushed, true);
            app.NoButton.FontName = 'Avenir';
            app.NoButton.Position = [378 58 100 24];
            app.NoButton.Text = 'No';

            % Create MarkasfiberLabel
            app.MarkasfiberLabel = uilabel(app.SortingAxesPanel);
            app.MarkasfiberLabel.HorizontalAlignment = 'center';
            app.MarkasfiberLabel.FontName = 'Avenir';
            app.MarkasfiberLabel.Position = [188 95 290 22];
            app.MarkasfiberLabel.Text = 'Mark as fiber?';

            % Create Toolbar
            app.Toolbar = uipanel(app.UIFigure);
            app.Toolbar.ForegroundColor = [1 1 1];
            app.Toolbar.BackgroundColor = [1 1 1];
            app.Toolbar.FontName = 'Avenir';
            app.Toolbar.Position = [25 675 1155 43];

            % Create ManualSegmentationButton
            app.ManualSegmentationButton = uibutton(app.Toolbar, 'push');
            app.ManualSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @ManualSegmentationButtonPushed, true);
            app.ManualSegmentationButton.BackgroundColor = [0.9608 0.9608 0.9608];
            app.ManualSegmentationButton.FontName = 'Avenir';
            app.ManualSegmentationButton.Enable = 'off';
            app.ManualSegmentationButton.Tooltip = {'Manually edit the fiber outlines generated in the "Initial Segmentation" stage as needed.'};
            app.ManualSegmentationButton.Position = [130 6 128 33];
            app.ManualSegmentationButton.Text = 'Manual Segmentation';

            % Create FiberPredictionButton
            app.FiberPredictionButton = uibutton(app.Toolbar, 'push');
            app.FiberPredictionButton.ButtonPushedFcn = createCallbackFcn(app, @FiberPredictionButtonPushed, true);
            app.FiberPredictionButton.BackgroundColor = [0.9608 0.9608 0.9608];
            app.FiberPredictionButton.FontName = 'Avenir';
            app.FiberPredictionButton.Enable = 'off';
            app.FiberPredictionButton.Tooltip = {'Predict fiber regions.'};
            app.FiberPredictionButton.Position = [262 6 120 33];
            app.FiberPredictionButton.Text = 'Fiber Prediction';

            % Create ManualFiberFilterButton
            app.ManualFiberFilterButton = uibutton(app.Toolbar, 'push');
            app.ManualFiberFilterButton.ButtonPushedFcn = createCallbackFcn(app, @ManualFiberFilterButtonPushed, true);
            app.ManualFiberFilterButton.FontName = 'Avenir';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualFiberFilterButton.Tooltip = {'Remove any image regions that were misclassified as fibers.'};
            app.ManualFiberFilterButton.Position = [387 6 120 33];
            app.ManualFiberFilterButton.Text = 'Manual Fiber Filter';

            % Create FiberPropertiesButton
            app.FiberPropertiesButton = uibutton(app.Toolbar, 'push');
            app.FiberPropertiesButton.ButtonPushedFcn = createCallbackFcn(app, @FiberPropertiesButtonPushed, true);
            app.FiberPropertiesButton.FontName = 'Avenir';
            app.FiberPropertiesButton.Enable = 'off';
            app.FiberPropertiesButton.Tooltip = {'Calculate minimum feret diameter and fiber area of fibers in the image.'};
            app.FiberPropertiesButton.Position = [511 6 120 33];
            app.FiberPropertiesButton.Text = 'Fiber Properties';

            % Create CentralNucleiButton
            app.CentralNucleiButton = uibutton(app.Toolbar, 'push');
            app.CentralNucleiButton.ButtonPushedFcn = createCallbackFcn(app, @CentralNucleiButtonPushed, true);
            app.CentralNucleiButton.FontName = 'Avenir';
            app.CentralNucleiButton.Enable = 'off';
            app.CentralNucleiButton.Tooltip = {'Mark fibers with centrally located nuclei.'};
            app.CentralNucleiButton.Position = [636 6 120 33];
            app.CentralNucleiButton.Text = 'Central Nuclei';

            % Create FiberTypingButton
            app.FiberTypingButton = uibutton(app.Toolbar, 'push');
            app.FiberTypingButton.ButtonPushedFcn = createCallbackFcn(app, @FiberTypingButtonPushed, true);
            app.FiberTypingButton.FontName = 'Avenir';
            app.FiberTypingButton.Enable = 'off';
            app.FiberTypingButton.Tooltip = {'Detect fibers with intensity of a certain color channel greater than a threshold value.'};
            app.FiberTypingButton.Position = [761 6 120 33];
            app.FiberTypingButton.Text = 'Fiber Typing';

            % Create NonfiberObjectsButton
            app.NonfiberObjectsButton = uibutton(app.Toolbar, 'push');
            app.NonfiberObjectsButton.ButtonPushedFcn = createCallbackFcn(app, @NonfiberObjectsButtonPushed, true);
            app.NonfiberObjectsButton.FontName = 'Avenir';
            app.NonfiberObjectsButton.Enable = 'off';
            app.NonfiberObjectsButton.Tooltip = {'Calculate and display objects that are not fibers.'};
            app.NonfiberObjectsButton.Position = [886 6 120 33];
            app.NonfiberObjectsButton.Text = 'Nonfiber Objects';

            % Create InitialSegmentationButton
            app.InitialSegmentationButton = uibutton(app.Toolbar, 'push');
            app.InitialSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @InitialSegmentationButtonPushed, true);
            app.InitialSegmentationButton.BackgroundColor = [0.9608 0.9608 0.9608];
            app.InitialSegmentationButton.FontName = 'Avenir';
            app.InitialSegmentationButton.Enable = 'off';
            app.InitialSegmentationButton.Tooltip = {'Segment the image to extract muscle fiber features, such as fiber boundaries, from the image.'};
            app.InitialSegmentationButton.Position = [6 6 120 33];
            app.InitialSegmentationButton.Text = 'Initial Segmentation';

            % Create NonfiberClassificationButton
            app.NonfiberClassificationButton = uibutton(app.Toolbar, 'push');
            app.NonfiberClassificationButton.ButtonPushedFcn = createCallbackFcn(app, @NonfiberClassificationButtonPushed, true);
            app.NonfiberClassificationButton.FontName = 'Avenir';
            app.NonfiberClassificationButton.Enable = 'off';
            app.NonfiberClassificationButton.Position = [1012 6 136 33];
            app.NonfiberClassificationButton.Text = 'Nonfiber Classification';

            % Create Panel
            app.Panel = uipanel(app.UIFigure);
            app.Panel.BackgroundColor = [0 0.2902 0.4784];
            app.Panel.FontName = 'Avenir';
            app.Panel.Position = [1 728 1199 52];

            % Create SMASHLabel
            app.SMASHLabel = uilabel(app.Panel);
            app.SMASHLabel.FontName = 'Avenir';
            app.SMASHLabel.FontSize = 24;
            app.SMASHLabel.FontColor = [1 1 1];
            app.SMASHLabel.Position = [56 10 87 35];
            app.SMASHLabel.Text = 'SMASH';

            % Create Panel_2
            app.Panel_2 = uipanel(app.Panel);
            app.Panel_2.ForegroundColor = [0 0.2784 0.4784];
            app.Panel_2.BorderType = 'none';
            app.Panel_2.BackgroundColor = [0.902 0.902 0.902];
            app.Panel_2.Position = [1085 9 102 35];

            % Create Image2
            app.Image2 = uiimage(app.Panel_2);
            app.Image2.HorizontalAlignment = 'right';
            app.Image2.VerticalAlignment = 'top';
            app.Image2.Position = [13 3 91 30];
            app.Image2.ImageSource = 'LabLogo.png';

            % Create Hyperlink
            app.Hyperlink = uihyperlink(app.Panel);
            app.Hyperlink.URL = 'https://sites.google.com/ucdavis.edu/myomatrixlab/home';
            app.Hyperlink.VisitedColor = [1 1 1];
            app.Hyperlink.FontName = 'Avenir';
            app.Hyperlink.FontSize = 14;
            app.Hyperlink.FontWeight = 'normal';
            app.Hyperlink.FontColor = [1 1 1];
            app.Hyperlink.Position = [1009 15 44 22];
            app.Hyperlink.Text = 'About';

            % Create Hyperlink_2
            app.Hyperlink_2 = uihyperlink(app.Panel);
            app.Hyperlink_2.URL = 'https://sites.google.com/ucdavis.edu/myomatrixlab/home';
            app.Hyperlink_2.VisitedColor = [1 1 1];
            app.Hyperlink_2.FontName = 'Avenir';
            app.Hyperlink_2.FontSize = 14;
            app.Hyperlink_2.FontWeight = 'normal';
            app.Hyperlink_2.FontColor = [1 1 1];
            app.Hyperlink_2.Position = [906 15 75 22];
            app.Hyperlink_2.Text = 'User Guide';

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.HorizontalAlignment = 'left';
            app.Image.Position = [10 728 36 55];
            app.Image.ImageSource = 'screenshot.png';

            % Create BatchModeLabel
            app.BatchModeLabel = uilabel(app.UIFigure);
            app.BatchModeLabel.FontName = 'Avenir';
            app.BatchModeLabel.Visible = 'off';
            app.BatchModeLabel.Position = [154 579 130 22];
            app.BatchModeLabel.Text = 'Batch Mode';

            % Create ImageBackground
            app.ImageBackground = uipanel(app.UIFigure);
            app.ImageBackground.Visible = 'off';
            app.ImageBackground.BackgroundColor = [0.9412 0.9412 0.9412];
            app.ImageBackground.Position = [331 28 827 625];

            % Create UIAxes
            app.UIAxes = uiaxes(app.ImageBackground);
            xlabel(app.UIAxes, 'X')
            ylabel(app.UIAxes, 'Y')
            app.UIAxes.FontName = 'Avenir';
            app.UIAxes.XColor = 'none';
            app.UIAxes.YColor = 'none';
            app.UIAxes.LineWidth = 1;
            app.UIAxes.Color = [0.9412 0.9412 0.9412];
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Box = 'on';
            app.UIAxes.Position = [7 -51 897 633];

            % Create SelectFileDescription
            app.SelectFileDescription = uilabel(app.UIFigure);
            app.SelectFileDescription.FontName = 'Avenir';
            app.SelectFileDescription.FontWeight = 'bold';
            app.SelectFileDescription.Position = [35 641 237 22];
            app.SelectFileDescription.Text = 'Please select an image to analyze:';

            % Create SelectFileDescription_2
            app.SelectFileDescription_2 = uilabel(app.UIFigure);
            app.SelectFileDescription_2.FontName = 'Avenir';
            app.SelectFileDescription_2.FontAngle = 'italic';
            app.SelectFileDescription_2.Position = [35 624 259 22];
            app.SelectFileDescription_2.Text = '*selecting multiple images enables batch mode';

            % Create ManualSegmentationControls
            app.ManualSegmentationControls = uipanel(app.UIFigure);
            app.ManualSegmentationControls.Visible = 'off';
            app.ManualSegmentationControls.BackgroundColor = [1 1 1];
            app.ManualSegmentationControls.FontName = 'Avenir';
            app.ManualSegmentationControls.FontWeight = 'bold';
            app.ManualSegmentationControls.Position = [19 113 298 459];

            % Create StartDrawingButton
            app.StartDrawingButton = uibutton(app.ManualSegmentationControls, 'push');
            app.StartDrawingButton.ButtonPushedFcn = createCallbackFcn(app, @StartDrawingButtonPushed, true);
            app.StartDrawingButton.FontName = 'Avenir';
            app.StartDrawingButton.Position = [28 281 100 24];
            app.StartDrawingButton.Text = 'Start Drawing';

            % Create AcceptLineButton
            app.AcceptLineButton = uibutton(app.ManualSegmentationControls, 'push');
            app.AcceptLineButton.ButtonPushedFcn = createCallbackFcn(app, @AcceptLineButtonPushed, true);
            app.AcceptLineButton.BackgroundColor = [0.9608 0.9608 0.9608];
            app.AcceptLineButton.FontName = 'Avenir';
            app.AcceptLineButton.Enable = 'off';
            app.AcceptLineButton.Position = [149 254 100 51];
            app.AcceptLineButton.Text = 'Accept Line';

            % Create CloseManualSegmentationButton
            app.CloseManualSegmentationButton = uibutton(app.ManualSegmentationControls, 'push');
            app.CloseManualSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @CloseManualSegmentationButtonPushed, true);
            app.CloseManualSegmentationButton.FontName = 'Avenir';
            app.CloseManualSegmentationButton.Position = [55 32 182 60];
            app.CloseManualSegmentationButton.Text = 'Close Manual Segmentation';

            % Create DrawingModeLabel
            app.DrawingModeLabel = uilabel(app.ManualSegmentationControls);
            app.DrawingModeLabel.FontName = 'Avenir';
            app.DrawingModeLabel.FontSize = 14;
            app.DrawingModeLabel.FontWeight = 'bold';
            app.DrawingModeLabel.Position = [11 364 103 22];
            app.DrawingModeLabel.Text = 'Drawing Mode';

            % Create MergeObjectsModeLabel
            app.MergeObjectsModeLabel = uilabel(app.ManualSegmentationControls);
            app.MergeObjectsModeLabel.FontName = 'Avenir';
            app.MergeObjectsModeLabel.FontSize = 14;
            app.MergeObjectsModeLabel.FontWeight = 'bold';
            app.MergeObjectsModeLabel.Position = [11 209 146 22];
            app.MergeObjectsModeLabel.Text = 'Merge Objects Mode';

            % Create StartMergingButton
            app.StartMergingButton = uibutton(app.ManualSegmentationControls, 'push');
            app.StartMergingButton.ButtonPushedFcn = createCallbackFcn(app, @StartMergingButtonPushed, true);
            app.StartMergingButton.FontName = 'Avenir';
            app.StartMergingButton.Position = [28 144 100 24];
            app.StartMergingButton.Text = 'Start Merging';

            % Create FinishDrawingButton
            app.FinishDrawingButton = uibutton(app.ManualSegmentationControls, 'push');
            app.FinishDrawingButton.ButtonPushedFcn = createCallbackFcn(app, @FinishDrawingButtonPushed, true);
            app.FinishDrawingButton.FontName = 'Avenir';
            app.FinishDrawingButton.Enable = 'off';
            app.FinishDrawingButton.Position = [28 252 100 24];
            app.FinishDrawingButton.Text = 'Finish Drawing';

            % Create ManualSegmentationDescription
            app.ManualSegmentationDescription = uilabel(app.ManualSegmentationControls);
            app.ManualSegmentationDescription.FontName = 'Avenir';
            app.ManualSegmentationDescription.FontWeight = 'bold';
            app.ManualSegmentationDescription.Position = [9 403 279 48];
            app.ManualSegmentationDescription.Text = {'This step allows you to manually edit the fiber'; 'outlines generated in the "Initial Segmentation"'; 'stage as needed.'};

            % Create ManualSegmentationDescription_2
            app.ManualSegmentationDescription_2 = uilabel(app.ManualSegmentationControls);
            app.ManualSegmentationDescription_2.FontName = 'Avenir';
            app.ManualSegmentationDescription_2.FontWeight = 'bold';
            app.ManualSegmentationDescription_2.Position = [6 98 293 22];
            app.ManualSegmentationDescription_2.Text = 'Once you are done editing the image, click below:';

            % Create ManualSegmentationDescription_3
            app.ManualSegmentationDescription_3 = uilabel(app.ManualSegmentationControls);
            app.ManualSegmentationDescription_3.FontName = 'Avenir';
            app.ManualSegmentationDescription_3.FontWeight = 'bold';
            app.ManualSegmentationDescription_3.Position = [19 315 277 48];
            app.ManualSegmentationDescription_3.Text = {'Select "Start Drawing" to begin drawing on the'; 'image. You may adjust the line and click'; '"Accept Line" once you are done.'};

            % Create ManualSegmentationDescription_4
            app.ManualSegmentationDescription_4 = uilabel(app.ManualSegmentationControls);
            app.ManualSegmentationDescription_4.FontName = 'Avenir';
            app.ManualSegmentationDescription_4.FontWeight = 'bold';
            app.ManualSegmentationDescription_4.Position = [19 178 242 32];
            app.ManualSegmentationDescription_4.Text = {'Select "Start Merging" to begin merging '; 'over-segmented regions.'};

            % Create FiberPredictionControlPanel
            app.FiberPredictionControlPanel = uipanel(app.UIFigure);
            app.FiberPredictionControlPanel.Visible = 'off';
            app.FiberPredictionControlPanel.BackgroundColor = [1 1 1];
            app.FiberPredictionControlPanel.FontName = 'Avenir';
            app.FiberPredictionControlPanel.Position = [29 170 276 394];

            % Create FilterButton
            app.FilterButton = uibutton(app.FiberPredictionControlPanel, 'push');
            app.FilterButton.ButtonPushedFcn = createCallbackFcn(app, @FilterButtonPushed, true);
            app.FilterButton.FontName = 'Avenir';
            app.FilterButton.Position = [74 135 100 24];
            app.FilterButton.Text = 'Filter';

            % Create ManualSortingButton
            app.ManualSortingButton = uibutton(app.FiberPredictionControlPanel, 'push');
            app.ManualSortingButton.ButtonPushedFcn = createCallbackFcn(app, @ManualSortingButtonPushed, true);
            app.ManualSortingButton.FontName = 'Avenir';
            app.ManualSortingButton.Enable = 'off';
            app.ManualSortingButton.Position = [74 39 100 24];
            app.ManualSortingButton.Text = 'Manual Sorting';

            % Create SortingThresholdHigherrequiresmoremanualsortingLabel
            app.SortingThresholdHigherrequiresmoremanualsortingLabel = uilabel(app.FiberPredictionControlPanel);
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.HorizontalAlignment = 'center';
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.FontName = 'Avenir';
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.Position = [-41 175 209 48];
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.Text = {'Sorting Threshold'; '(Higher requires'; ' more manual sorting)'; ''};

            % Create SortingThresholdSlider
            app.SortingThresholdSlider = uislider(app.FiberPredictionControlPanel);
            app.SortingThresholdSlider.Limits = [0 0.9];
            app.SortingThresholdSlider.MajorTicks = [0 0.5 0.9];
            app.SortingThresholdSlider.ValueChangedFcn = createCallbackFcn(app, @SortingThresholdSliderValueChanged, true);
            app.SortingThresholdSlider.MinorTicks = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
            app.SortingThresholdSlider.FontName = 'Avenir';
            app.SortingThresholdSlider.Position = [145 209 104 3];

            % Create PizelSizeumpixelLabel
            app.PizelSizeumpixelLabel = uilabel(app.FiberPredictionControlPanel);
            app.PizelSizeumpixelLabel.HorizontalAlignment = 'right';
            app.PizelSizeumpixelLabel.Position = [16 239 114 22];
            app.PizelSizeumpixelLabel.Text = 'Pizel Size (um/pixel)';

            % Create PixelSizeFiberPrediction
            app.PixelSizeFiberPrediction = uieditfield(app.FiberPredictionControlPanel, 'numeric');
            app.PixelSizeFiberPrediction.Limits = [0 Inf];
            app.PixelSizeFiberPrediction.ValueChangedFcn = createCallbackFcn(app, @PixelSizeFiberPredictionValueChanged, true);
            app.PixelSizeFiberPrediction.Position = [136 239 100 22];

            % Create FiberPredictionDescription
            app.FiberPredictionDescription = uilabel(app.FiberPredictionControlPanel);
            app.FiberPredictionDescription.FontWeight = 'bold';
            app.FiberPredictionDescription.Position = [9 357 240 28];
            app.FiberPredictionDescription.Text = {'This stage predicts which regions of the '; 'image are fibers.'};

            % Create FiberPredictionDescription_2
            app.FiberPredictionDescription_2 = uilabel(app.FiberPredictionControlPanel);
            app.FiberPredictionDescription_2.FontWeight = 'bold';
            app.FiberPredictionDescription_2.Position = [9 276 269 56];
            app.FiberPredictionDescription_2.Text = {'Specify the Pixel Size and Sorting Threshold. '; 'Note that the sorting threshold is the '; 'probability that something is a fiber or'; 'non-fiber.'};

            % Create FiberPredictionDescription_3
            app.FiberPredictionDescription_3 = uilabel(app.FiberPredictionControlPanel);
            app.FiberPredictionDescription_3.FontWeight = 'bold';
            app.FiberPredictionDescription_3.Position = [9 72 253 28];
            app.FiberPredictionDescription_3.Text = {'Manually sort the regions that could not be'; 'classified:'};

            % Create SegmentationParameters
            app.SegmentationParameters = uipanel(app.UIFigure);
            app.SegmentationParameters.Visible = 'off';
            app.SegmentationParameters.BackgroundColor = [1 1 1];
            app.SegmentationParameters.FontName = 'Avenir';
            app.SegmentationParameters.FontSize = 14;
            app.SegmentationParameters.Position = [22 168 288 399];

            % Create FiberOutlineChannelColorBox
            app.FiberOutlineChannelColorBox = uiaxes(app.SegmentationParameters);
            app.FiberOutlineChannelColorBox.Toolbar.Visible = 'off';
            app.FiberOutlineChannelColorBox.FontName = 'Avenir';
            app.FiberOutlineChannelColorBox.XTick = [];
            app.FiberOutlineChannelColorBox.YTick = [];
            app.FiberOutlineChannelColorBox.Color = [0 1 1];
            app.FiberOutlineChannelColorBox.Box = 'on';
            app.FiberOutlineChannelColorBox.PickableParts = 'none';
            app.FiberOutlineChannelColorBox.Position = [238 224 30 30];

            % Create SegmentButton
            app.SegmentButton = uibutton(app.SegmentationParameters, 'push');
            app.SegmentButton.ButtonPushedFcn = createCallbackFcn(app, @SegmentButtonPushed, true);
            app.SegmentButton.FontName = 'Avenir';
            app.SegmentButton.Position = [152 122 100 24];
            app.SegmentButton.Text = 'Segment';

            % Create FiberOutlineColorDropDownLabel
            app.FiberOutlineColorDropDownLabel = uilabel(app.SegmentationParameters);
            app.FiberOutlineColorDropDownLabel.HorizontalAlignment = 'right';
            app.FiberOutlineColorDropDownLabel.FontName = 'Avenir';
            app.FiberOutlineColorDropDownLabel.Position = [6 228 109 22];
            app.FiberOutlineColorDropDownLabel.Text = 'Fiber Outline Color';

            % Create FiberOutlineColorDropDown
            app.FiberOutlineColorDropDown = uidropdown(app.SegmentationParameters);
            app.FiberOutlineColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.FiberOutlineColorDropDown.ItemsData = {'1', '2', '3'};
            app.FiberOutlineColorDropDown.ValueChangedFcn = createCallbackFcn(app, @FiberOutlineColorValueChanged, true);
            app.FiberOutlineColorDropDown.FontName = 'Avenir';
            app.FiberOutlineColorDropDown.Position = [126 228 106 22];
            app.FiberOutlineColorDropDown.Value = '1';

            % Create AcceptSegmentationButton
            app.AcceptSegmentationButton = uibutton(app.SegmentationParameters, 'push');
            app.AcceptSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @AcceptSegmentationButtonPushed, true);
            app.AcceptSegmentationButton.FontName = 'Avenir';
            app.AcceptSegmentationButton.Enable = 'off';
            app.AcceptSegmentationButton.Position = [89 75 100 24];
            app.AcceptSegmentationButton.Text = 'Accept';

            % Create SegmentationThresholdSliderLabel
            app.SegmentationThresholdSliderLabel = uilabel(app.SegmentationParameters);
            app.SegmentationThresholdSliderLabel.HorizontalAlignment = 'center';
            app.SegmentationThresholdSliderLabel.FontName = 'Avenir';
            app.SegmentationThresholdSliderLabel.Position = [10 168 80 43];
            app.SegmentationThresholdSliderLabel.Text = {'Segmentation'; 'Threshold'};

            % Create SegmentationThresholdSlider
            app.SegmentationThresholdSlider = uislider(app.SegmentationParameters);
            app.SegmentationThresholdSlider.Limits = [0 50];
            app.SegmentationThresholdSlider.ValueChangedFcn = createCallbackFcn(app, @SegmentationThresholdSliderValueChanged, true);
            app.SegmentationThresholdSlider.FontName = 'Avenir';
            app.SegmentationThresholdSlider.Position = [104 194 153 3];

            % Create DetectValueButton
            app.DetectValueButton = uibutton(app.SegmentationParameters, 'push');
            app.DetectValueButton.ButtonPushedFcn = createCallbackFcn(app, @DetectValueButtonPushed, true);
            app.DetectValueButton.BackgroundColor = [0.902 0.902 0.902];
            app.DetectValueButton.FontName = 'Avenir';
            app.DetectValueButton.Position = [27 122 100 24];
            app.DetectValueButton.Text = 'Detect Value';

            % Create InitialSegmentationDescription
            app.InitialSegmentationDescription = uilabel(app.SegmentationParameters);
            app.InitialSegmentationDescription.FontName = 'Avenir';
            app.InitialSegmentationDescription.FontWeight = 'bold';
            app.InitialSegmentationDescription.Position = [15 354 237 39];
            app.InitialSegmentationDescription.Text = {'This step segments the image to extract'; 'muscle fiber features from the image.'};

            % Create InitialSegmentationDirections
            app.InitialSegmentationDirections = uilabel(app.SegmentationParameters);
            app.InitialSegmentationDirections.FontName = 'Avenir';
            app.InitialSegmentationDirections.FontWeight = 'bold';
            app.InitialSegmentationDirections.Position = [15 259 267 84];
            app.InitialSegmentationDirections.Text = {'Adjust the Fiber Outline Color, Pixel Size, '; 'and Segmentation Threshold to the desired '; 'values. Optionally, “Detect Value” can be'; 'used to predict a segmentation value. Select'; '“Segment” once the desired values are '; 'chosen, and click "Accept" to save changes.'};

            % Create InitialSegmentationDescription_2
            app.InitialSegmentationDescription_2 = uilabel(app.SegmentationParameters);
            app.InitialSegmentationDescription_2.FontName = 'Avenir';
            app.InitialSegmentationDescription_2.FontSize = 14;
            app.InitialSegmentationDescription_2.FontWeight = 'bold';
            app.InitialSegmentationDescription_2.Visible = 'off';
            app.InitialSegmentationDescription_2.Position = [17 67 287 39];
            app.InitialSegmentationDescription_2.Text = {'Select "Start Merging" to begin merging '; 'over-segmented regions.'};

            % Create CloseInitialSegmentationButton
            app.CloseInitialSegmentationButton = uibutton(app.SegmentationParameters, 'push');
            app.CloseInitialSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @CloseInitialSegmentationButtonPushed, true);
            app.CloseInitialSegmentationButton.FontName = 'Avenir';
            app.CloseInitialSegmentationButton.Position = [89 38 100 24];
            app.CloseInitialSegmentationButton.Text = 'Close';

            % Create ManualFilterControls
            app.ManualFilterControls = uipanel(app.UIFigure);
            app.ManualFilterControls.Visible = 'off';
            app.ManualFilterControls.BackgroundColor = [1 1 1];
            app.ManualFilterControls.FontName = 'Avenir';
            app.ManualFilterControls.Position = [29 306 259 255];

            % Create RemoveNonfibersButton
            app.RemoveNonfibersButton = uibutton(app.ManualFilterControls, 'push');
            app.RemoveNonfibersButton.ButtonPushedFcn = createCallbackFcn(app, @RemoveNonfibersButtonPushed, true);
            app.RemoveNonfibersButton.FontName = 'Avenir';
            app.RemoveNonfibersButton.Position = [68 94 115 24];
            app.RemoveNonfibersButton.Text = 'Remove Nonfibers';

            % Create FinishManualFilteringButton
            app.FinishManualFilteringButton = uibutton(app.ManualFilterControls, 'push');
            app.FinishManualFilteringButton.ButtonPushedFcn = createCallbackFcn(app, @FinishManualFilteringButtonPushed, true);
            app.FinishManualFilteringButton.FontName = 'Avenir';
            app.FinishManualFilteringButton.Enable = 'off';
            app.FinishManualFilteringButton.Position = [58 53 136 24];
            app.FinishManualFilteringButton.Text = 'Finish Manual Filtering';

            % Create ManualFilterDescription
            app.ManualFilterDescription = uilabel(app.ManualFilterControls);
            app.ManualFilterDescription.FontWeight = 'bold';
            app.ManualFilterDescription.Position = [9 220 221 28];
            app.ManualFilterDescription.Text = {'Remove any image regions that were '; 'misclassified as fibers.'};

            % Create ManualFilterDescription_2
            app.ManualFilterDescription_2 = uilabel(app.ManualFilterControls);
            app.ManualFilterDescription_2.FontWeight = 'bold';
            app.ManualFilterDescription_2.Position = [9 137 259 56];
            app.ManualFilterDescription_2.Text = {'Click “Remove Nonfibers” and click on any '; 'misclassified fibers to remove them. If you '; 'accidentally remove a correctly classified'; 'fiber, click again to undo. '};

            % Create FiberPropertiesControlPanel
            app.FiberPropertiesControlPanel = uipanel(app.UIFigure);
            app.FiberPropertiesControlPanel.Visible = 'off';
            app.FiberPropertiesControlPanel.BackgroundColor = [1 1 1];
            app.FiberPropertiesControlPanel.FontName = 'Avenir';
            app.FiberPropertiesControlPanel.Position = [31 224 250 337];

            % Create WritetoExcelButton
            app.WritetoExcelButton = uibutton(app.FiberPropertiesControlPanel, 'push');
            app.WritetoExcelButton.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelButtonPushed, true);
            app.WritetoExcelButton.FontName = 'Avenir';
            app.WritetoExcelButton.Enable = 'off';
            app.WritetoExcelButton.Position = [136 105 100 24];
            app.WritetoExcelButton.Text = 'Write to Excel';

            % Create DoneFiberProperties
            app.DoneFiberProperties = uibutton(app.FiberPropertiesControlPanel, 'push');
            app.DoneFiberProperties.ButtonPushedFcn = createCallbackFcn(app, @DoneFiberPropertiesPushed, true);
            app.DoneFiberProperties.FontName = 'Avenir';
            app.DoneFiberProperties.Position = [79 48 100 24];
            app.DoneFiberProperties.Text = 'Close';

            % Create DataOutputFolderEditFieldLabel
            app.DataOutputFolderEditFieldLabel = uilabel(app.FiberPropertiesControlPanel);
            app.DataOutputFolderEditFieldLabel.HorizontalAlignment = 'right';
            app.DataOutputFolderEditFieldLabel.FontName = 'Avenir';
            app.DataOutputFolderEditFieldLabel.Position = [13 145 111 22];
            app.DataOutputFolderEditFieldLabel.Text = 'Data Output Folder';

            % Create FiberPropertiesDataOutputFolder
            app.FiberPropertiesDataOutputFolder = uieditfield(app.FiberPropertiesControlPanel, 'text');
            app.FiberPropertiesDataOutputFolder.FontName = 'Avenir';
            app.FiberPropertiesDataOutputFolder.Position = [139 145 100 22];

            % Create PixelSizeumpixelEditFieldLabel_2
            app.PixelSizeumpixelEditFieldLabel_2 = uilabel(app.FiberPropertiesControlPanel);
            app.PixelSizeumpixelEditFieldLabel_2.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditFieldLabel_2.FontName = 'Avenir';
            app.PixelSizeumpixelEditFieldLabel_2.FontWeight = 'bold';
            app.PixelSizeumpixelEditFieldLabel_2.Visible = 'off';
            app.PixelSizeumpixelEditFieldLabel_2.Position = [3 177 121 22];
            app.PixelSizeumpixelEditFieldLabel_2.Text = 'Pixel Size (um/pixel)';

            % Create PixelSizeFiberProperties
            app.PixelSizeFiberProperties = uieditfield(app.FiberPropertiesControlPanel, 'numeric');
            app.PixelSizeFiberProperties.Limits = [0 Inf];
            app.PixelSizeFiberProperties.ValueChangedFcn = createCallbackFcn(app, @PixelSizeFiberPropertiesValueChanged, true);
            app.PixelSizeFiberProperties.FontName = 'Avenir';
            app.PixelSizeFiberProperties.FontWeight = 'bold';
            app.PixelSizeFiberProperties.Visible = 'off';
            app.PixelSizeFiberProperties.Position = [9 340 35 22];

            % Create CalculateFiberProperties
            app.CalculateFiberProperties = uibutton(app.FiberPropertiesControlPanel, 'push');
            app.CalculateFiberProperties.ButtonPushedFcn = createCallbackFcn(app, @CalculateFiberPropertiesPushed, true);
            app.CalculateFiberProperties.FontName = 'Avenir';
            app.CalculateFiberProperties.Position = [21 105 100 24];
            app.CalculateFiberProperties.Text = 'Calculate';

            % Create FiberPropertiesDescription
            app.FiberPropertiesDescription = uilabel(app.FiberPropertiesControlPanel);
            app.FiberPropertiesDescription.FontWeight = 'bold';
            app.FiberPropertiesDescription.Position = [11 298 228 28];
            app.FiberPropertiesDescription.Text = {'Calculate minimum feret diameter and '; 'fiber area of fibers in the image.'};

            % Create FiberPropertiesDescription_2
            app.FiberPropertiesDescription_2 = uilabel(app.FiberPropertiesControlPanel);
            app.FiberPropertiesDescription_2.FontWeight = 'bold';
            app.FiberPropertiesDescription_2.Position = [11 209 243 56];
            app.FiberPropertiesDescription_2.Text = {'Input the Pixel Size and Data Output'; 'Folder. Press "Calculate" to run calculate'; 'the fiber properties, and "Write to Excel"'; 'to save the data to Excel.'};

            % Create FiberPropertiesFileWriteStatusLabel
            app.FiberPropertiesFileWriteStatusLabel = uilabel(app.FiberPropertiesControlPanel);
            app.FiberPropertiesFileWriteStatusLabel.Position = [26 77 211 22];
            app.FiberPropertiesFileWriteStatusLabel.Text = 'Fiber Properties File Write Status';

            % Create NonfiberObjectsControlPanel
            app.NonfiberObjectsControlPanel = uipanel(app.UIFigure);
            app.NonfiberObjectsControlPanel.Visible = 'off';
            app.NonfiberObjectsControlPanel.BackgroundColor = [1 1 1];
            app.NonfiberObjectsControlPanel.FontName = 'Avenir';
            app.NonfiberObjectsControlPanel.Position = [14 131 260 424];

            % Create NonfiberChannelColorBox
            app.NonfiberChannelColorBox = uiaxes(app.NonfiberObjectsControlPanel);
            app.NonfiberChannelColorBox.Toolbar.Visible = 'off';
            app.NonfiberChannelColorBox.FontName = 'Avenir';
            app.NonfiberChannelColorBox.XTick = [];
            app.NonfiberChannelColorBox.YTick = [];
            app.NonfiberChannelColorBox.Color = [0 1 1];
            app.NonfiberChannelColorBox.Box = 'on';
            app.NonfiberChannelColorBox.Visible = 'off';
            app.NonfiberChannelColorBox.PickableParts = 'none';
            app.NonfiberChannelColorBox.Position = [227 206 30 30];

            % Create ObjectColorDropDownLabel
            app.ObjectColorDropDownLabel = uilabel(app.NonfiberObjectsControlPanel);
            app.ObjectColorDropDownLabel.HorizontalAlignment = 'right';
            app.ObjectColorDropDownLabel.FontName = 'Avenir';
            app.ObjectColorDropDownLabel.Tooltip = {'Based on the channel used for imaging'};
            app.ObjectColorDropDownLabel.Position = [33 210 75 22];
            app.ObjectColorDropDownLabel.Text = 'Object Color';

            % Create NonfiberObjectsColorDropDown
            app.NonfiberObjectsColorDropDown = uidropdown(app.NonfiberObjectsControlPanel);
            app.NonfiberObjectsColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.NonfiberObjectsColorDropDown.ItemsData = {'1', '2', '3'};
            app.NonfiberObjectsColorDropDown.ValueChangedFcn = createCallbackFcn(app, @NonfiberObjectsColorDropDownValueChanged, true);
            app.NonfiberObjectsColorDropDown.FontName = 'Avenir';
            app.NonfiberObjectsColorDropDown.Position = [123 210 100 22];
            app.NonfiberObjectsColorDropDown.Value = '1';

            % Create DataOutputFolderEditField_3Label_2
            app.DataOutputFolderEditField_3Label_2 = uilabel(app.NonfiberObjectsControlPanel);
            app.DataOutputFolderEditField_3Label_2.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_3Label_2.FontName = 'Avenir';
            app.DataOutputFolderEditField_3Label_2.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.DataOutputFolderEditField_3Label_2.Position = [2 172 111 22];
            app.DataOutputFolderEditField_3Label_2.Text = 'Data Output Folder';

            % Create NonfiberObjectsDataOutputFolder
            app.NonfiberObjectsDataOutputFolder = uieditfield(app.NonfiberObjectsControlPanel, 'text');
            app.NonfiberObjectsDataOutputFolder.FontName = 'Avenir';
            app.NonfiberObjectsDataOutputFolder.Position = [128 172 100 22];

            % Create CalculateNonfiberObjects
            app.CalculateNonfiberObjects = uibutton(app.NonfiberObjectsControlPanel, 'push');
            app.CalculateNonfiberObjects.ButtonPushedFcn = createCallbackFcn(app, @CalculateNonfiberObjectsButtonPushed, true);
            app.CalculateNonfiberObjects.FontName = 'Avenir';
            app.CalculateNonfiberObjects.Position = [22 120 100 24];
            app.CalculateNonfiberObjects.Text = 'Calculate';

            % Create WritetoExcelNonfiber
            app.WritetoExcelNonfiber = uibutton(app.NonfiberObjectsControlPanel, 'push');
            app.WritetoExcelNonfiber.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelNonfiberButtonPushed, true);
            app.WritetoExcelNonfiber.FontName = 'Avenir';
            app.WritetoExcelNonfiber.Position = [137 120 100 24];
            app.WritetoExcelNonfiber.Text = 'Write to Excel';

            % Create DoneNonfiber
            app.DoneNonfiber = uibutton(app.NonfiberObjectsControlPanel, 'push');
            app.DoneNonfiber.ButtonPushedFcn = createCallbackFcn(app, @DoneNonfiberButtonPushed, true);
            app.DoneNonfiber.FontName = 'Avenir';
            app.DoneNonfiber.Position = [83 56 100 24];
            app.DoneNonfiber.Text = 'Close';

            % Create PixelSizeumpixelLabel_2
            app.PixelSizeumpixelLabel_2 = uilabel(app.NonfiberObjectsControlPanel);
            app.PixelSizeumpixelLabel_2.HorizontalAlignment = 'right';
            app.PixelSizeumpixelLabel_2.FontName = 'Avenir';
            app.PixelSizeumpixelLabel_2.Tooltip = {'Pixel size is default based on microscope, adjust pixel size to your specific microscope.'};
            app.PixelSizeumpixelLabel_2.Position = [51 245 59 32];
            app.PixelSizeumpixelLabel_2.Text = {'Pixel Size'; '(um/pixel)'};

            % Create PixelSizeNonfiberObjects
            app.PixelSizeNonfiberObjects = uieditfield(app.NonfiberObjectsControlPanel, 'numeric');
            app.PixelSizeNonfiberObjects.Limits = [0 Inf];
            app.PixelSizeNonfiberObjects.ValueChangedFcn = createCallbackFcn(app, @PixelSizeNonfiberObjectsValueChanged, true);
            app.PixelSizeNonfiberObjects.FontName = 'Avenir';
            app.PixelSizeNonfiberObjects.Position = [125 255 100 22];

            % Create NonfiberObjectsFileWriteStatusLabel
            app.NonfiberObjectsFileWriteStatusLabel = uilabel(app.NonfiberObjectsControlPanel);
            app.NonfiberObjectsFileWriteStatusLabel.Position = [36 92 186 22];
            app.NonfiberObjectsFileWriteStatusLabel.Text = 'Nonfiber Objects File Write Status';

            % Create NonfiberObjectsDescription
            app.NonfiberObjectsDescription = uilabel(app.NonfiberObjectsControlPanel);
            app.NonfiberObjectsDescription.FontWeight = 'bold';
            app.NonfiberObjectsDescription.Position = [8 385 226 28];
            app.NonfiberObjectsDescription.Text = {'Calculate and display objects that are '; 'not fibers. '};

            % Create NonfiberObjectsDescription_2
            app.NonfiberObjectsDescription_2 = uilabel(app.NonfiberObjectsControlPanel);
            app.NonfiberObjectsDescription_2.FontWeight = 'bold';
            app.NonfiberObjectsDescription_2.Position = [8 297 247 70];
            app.NonfiberObjectsDescription_2.Text = {'Set the field values below. Hover over the '; 'field names for more information. Press '; '"Calculate" to calculate nonfiber objects,'; 'and "Write to Excel" to save the '; 'data to Excel.'};

            % Create CentralNucleiControlPanel
            app.CentralNucleiControlPanel = uipanel(app.UIFigure);
            app.CentralNucleiControlPanel.Visible = 'off';
            app.CentralNucleiControlPanel.BackgroundColor = [1 1 1];
            app.CentralNucleiControlPanel.FontName = 'Avenir';
            app.CentralNucleiControlPanel.Position = [16 120 264 446];

            % Create CentralNucleiChannelColorBox
            app.CentralNucleiChannelColorBox = uiaxes(app.CentralNucleiControlPanel);
            app.CentralNucleiChannelColorBox.Toolbar.Visible = 'off';
            app.CentralNucleiChannelColorBox.FontName = 'Avenir';
            app.CentralNucleiChannelColorBox.XTick = [];
            app.CentralNucleiChannelColorBox.YTick = [];
            app.CentralNucleiChannelColorBox.Color = [0 1 1];
            app.CentralNucleiChannelColorBox.Box = 'on';
            app.CentralNucleiChannelColorBox.Visible = 'off';
            app.CentralNucleiChannelColorBox.PickableParts = 'none';
            app.CentralNucleiChannelColorBox.Position = [228 237 30 30];

            % Create PixelSizeumpixelLabel
            app.PixelSizeumpixelLabel = uilabel(app.CentralNucleiControlPanel);
            app.PixelSizeumpixelLabel.HorizontalAlignment = 'right';
            app.PixelSizeumpixelLabel.FontName = 'Avenir';
            app.PixelSizeumpixelLabel.Tooltip = {'Pixel size is default based on microscope, adjust pixel size to your specific microscope.'};
            app.PixelSizeumpixelLabel.Position = [48 265 58 32];
            app.PixelSizeumpixelLabel.Text = {'Pixel Size'; 'um/pixel'};

            % Create PixelSizeCentralNuclei
            app.PixelSizeCentralNuclei = uieditfield(app.CentralNucleiControlPanel, 'numeric');
            app.PixelSizeCentralNuclei.Limits = [0 Inf];
            app.PixelSizeCentralNuclei.ValueChangedFcn = createCallbackFcn(app, @PixelSizeCentralNucleiValueChanged, true);
            app.PixelSizeCentralNuclei.FontName = 'Avenir';
            app.PixelSizeCentralNuclei.Tooltip = {'Minimum size for a region to be considered a nucleus.'};
            app.PixelSizeCentralNuclei.Position = [121 275 100 22];

            % Create NucleiColorDropDownLabel
            app.NucleiColorDropDownLabel = uilabel(app.CentralNucleiControlPanel);
            app.NucleiColorDropDownLabel.HorizontalAlignment = 'right';
            app.NucleiColorDropDownLabel.FontName = 'Avenir';
            app.NucleiColorDropDownLabel.Tooltip = {'Based on the channel used for imaging'};
            app.NucleiColorDropDownLabel.Position = [37 240 73 22];
            app.NucleiColorDropDownLabel.Text = 'Nuclei Color';

            % Create NucleiColorDropDown
            app.NucleiColorDropDown = uidropdown(app.CentralNucleiControlPanel);
            app.NucleiColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.NucleiColorDropDown.ItemsData = {'1', '2', '3'};
            app.NucleiColorDropDown.ValueChangedFcn = createCallbackFcn(app, @NucleiColorValueChanged, true);
            app.NucleiColorDropDown.Tooltip = {'Minimum size for a region to be considered a nucleus.'};
            app.NucleiColorDropDown.FontName = 'Avenir';
            app.NucleiColorDropDown.Position = [125 240 100 22];
            app.NucleiColorDropDown.Value = '1';

            % Create CalculateCentralNuclei
            app.CalculateCentralNuclei = uibutton(app.CentralNucleiControlPanel, 'push');
            app.CalculateCentralNuclei.ButtonPushedFcn = createCallbackFcn(app, @CalculateCentralNucleiPushed, true);
            app.CalculateCentralNuclei.FontName = 'Avenir';
            app.CalculateCentralNuclei.Position = [26 95 100 24];
            app.CalculateCentralNuclei.Text = 'Calculate';

            % Create CentralNucleiExcelWrite
            app.CentralNucleiExcelWrite = uibutton(app.CentralNucleiControlPanel, 'push');
            app.CentralNucleiExcelWrite.ButtonPushedFcn = createCallbackFcn(app, @CentralNucleiExcelWriteButtonPushed, true);
            app.CentralNucleiExcelWrite.FontName = 'Avenir';
            app.CentralNucleiExcelWrite.Position = [138 95 100 24];
            app.CentralNucleiExcelWrite.Text = 'Write To Excel';

            % Create DoneCentralNuclei
            app.DoneCentralNuclei = uibutton(app.CentralNucleiControlPanel, 'push');
            app.DoneCentralNuclei.ButtonPushedFcn = createCallbackFcn(app, @DoneCentralNucleiPushed, true);
            app.DoneCentralNuclei.FontName = 'Avenir';
            app.DoneCentralNuclei.Position = [79 42 100 24];
            app.DoneCentralNuclei.Text = 'Close';

            % Create DistancefromborderEditFieldLabel
            app.DistancefromborderEditFieldLabel = uilabel(app.CentralNucleiControlPanel);
            app.DistancefromborderEditFieldLabel.HorizontalAlignment = 'right';
            app.DistancefromborderEditFieldLabel.FontName = 'Avenir';
            app.DistancefromborderEditFieldLabel.Tooltip = {'Distance of the nuclei from the fiber border.'};
            app.DistancefromborderEditFieldLabel.Position = [14 208 119 22];
            app.DistancefromborderEditFieldLabel.Text = 'Distance from border';

            % Create DistancefromborderEditField
            app.DistancefromborderEditField = uieditfield(app.CentralNucleiControlPanel, 'numeric');
            app.DistancefromborderEditField.Limits = [0 Inf];
            app.DistancefromborderEditField.FontName = 'Avenir';
            app.DistancefromborderEditField.Position = [149 208 100 22];

            % Create MinimumNucleusSizeum2EditFieldLabel
            app.MinimumNucleusSizeum2EditFieldLabel = uilabel(app.CentralNucleiControlPanel);
            app.MinimumNucleusSizeum2EditFieldLabel.HorizontalAlignment = 'right';
            app.MinimumNucleusSizeum2EditFieldLabel.FontName = 'Avenir';
            app.MinimumNucleusSizeum2EditFieldLabel.Visible = 'off';
            app.MinimumNucleusSizeum2EditFieldLabel.Tooltip = {'Minimum size for a region to be considered a nucleus.'};
            app.MinimumNucleusSizeum2EditFieldLabel.Position = [-1 167 136 32];
            app.MinimumNucleusSizeum2EditFieldLabel.Text = {'Minimum Nucleus Size'; '(um^2)'};

            % Create MinimumNucleusSizeum2EditField
            app.MinimumNucleusSizeum2EditField = uieditfield(app.CentralNucleiControlPanel, 'numeric');
            app.MinimumNucleusSizeum2EditField.Limits = [0 Inf];
            app.MinimumNucleusSizeum2EditField.FontName = 'Avenir';
            app.MinimumNucleusSizeum2EditField.Visible = 'off';
            app.MinimumNucleusSizeum2EditField.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.MinimumNucleusSizeum2EditField.Position = [150 177 100 22];

            % Create DataOutputFolderEditField_2Label
            app.DataOutputFolderEditField_2Label = uilabel(app.CentralNucleiControlPanel);
            app.DataOutputFolderEditField_2Label.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_2Label.FontName = 'Avenir';
            app.DataOutputFolderEditField_2Label.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.DataOutputFolderEditField_2Label.Position = [13 142 111 22];
            app.DataOutputFolderEditField_2Label.Text = 'Data Output Folder';

            % Create CentralNucleiDataOutputFolder
            app.CentralNucleiDataOutputFolder = uieditfield(app.CentralNucleiControlPanel, 'text');
            app.CentralNucleiDataOutputFolder.FontName = 'Avenir';
            app.CentralNucleiDataOutputFolder.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.CentralNucleiDataOutputFolder.Position = [142 142 109 22];

            % Create CentralNucleiDescription
            app.CentralNucleiDescription = uilabel(app.CentralNucleiControlPanel);
            app.CentralNucleiDescription.FontWeight = 'bold';
            app.CentralNucleiDescription.Position = [9 407 238 22];
            app.CentralNucleiDescription.Text = {'Mark fibers with centrally located nuclei.'; ''};

            % Create CentralNucleiDescription_2
            app.CentralNucleiDescription_2 = uilabel(app.CentralNucleiControlPanel);
            app.CentralNucleiDescription_2.FontWeight = 'bold';
            app.CentralNucleiDescription_2.Position = [9 310 247 70];
            app.CentralNucleiDescription_2.Text = {'Set the field values below. Hover over the '; 'field names for more information. Press '; '"Calculate" to calculate the central nuclei,'; 'and "Write to Excel" to save the '; 'data to Excel.'; ''};

            % Create CentralNucleiFileWriteStatusLabel
            app.CentralNucleiFileWriteStatusLabel = uilabel(app.CentralNucleiControlPanel);
            app.CentralNucleiFileWriteStatusLabel.Position = [38 70 172 22];
            app.CentralNucleiFileWriteStatusLabel.Text = 'Central Nuclei File Write Status';

            % Create NonfiberClassificationControlPanel
            app.NonfiberClassificationControlPanel = uipanel(app.UIFigure);
            app.NonfiberClassificationControlPanel.Visible = 'off';
            app.NonfiberClassificationControlPanel.BackgroundColor = [1 1 1];
            app.NonfiberClassificationControlPanel.FontName = 'Avenir';
            app.NonfiberClassificationControlPanel.Position = [13 131 260 430];

            % Create NonfiberClassificationChannelColorBox
            app.NonfiberClassificationChannelColorBox = uiaxes(app.NonfiberClassificationControlPanel);
            app.NonfiberClassificationChannelColorBox.Toolbar.Visible = 'off';
            app.NonfiberClassificationChannelColorBox.FontName = 'Avenir';
            app.NonfiberClassificationChannelColorBox.XTick = [];
            app.NonfiberClassificationChannelColorBox.YTick = [];
            app.NonfiberClassificationChannelColorBox.Color = [0 1 1];
            app.NonfiberClassificationChannelColorBox.Box = 'on';
            app.NonfiberClassificationChannelColorBox.Position = [228 185 30 30];

            % Create PixelSizeumpixelEditField_3Label_2
            app.PixelSizeumpixelEditField_3Label_2 = uilabel(app.NonfiberClassificationControlPanel);
            app.PixelSizeumpixelEditField_3Label_2.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditField_3Label_2.FontName = 'Avenir';
            app.PixelSizeumpixelEditField_3Label_2.Tooltip = {'Pixel size is default based on microscope, adjust pixel size to your specific microscope.'};
            app.PixelSizeumpixelEditField_3Label_2.Position = [44 217 59 32];
            app.PixelSizeumpixelEditField_3Label_2.Text = {'Pixel Size'; '(um/pixel)'};

            % Create PixelSizeNonfiberClassification
            app.PixelSizeNonfiberClassification = uieditfield(app.NonfiberClassificationControlPanel, 'numeric');
            app.PixelSizeNonfiberClassification.Limits = [0 Inf];
            app.PixelSizeNonfiberClassification.ValueChangedFcn = createCallbackFcn(app, @PixelSizeNonfiberClassificationValueChanged, true);
            app.PixelSizeNonfiberClassification.FontName = 'Avenir';
            app.PixelSizeNonfiberClassification.Position = [118 227 100 22];

            % Create DataOutputFolderEditField_3Label_3
            app.DataOutputFolderEditField_3Label_3 = uilabel(app.NonfiberClassificationControlPanel);
            app.DataOutputFolderEditField_3Label_3.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_3Label_3.FontName = 'Avenir';
            app.DataOutputFolderEditField_3Label_3.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.DataOutputFolderEditField_3Label_3.Position = [16 155 111 22];
            app.DataOutputFolderEditField_3Label_3.Text = 'Data Output Folder';

            % Create NonfiberClassificationDataOutputFolder
            app.NonfiberClassificationDataOutputFolder = uieditfield(app.NonfiberClassificationControlPanel, 'text');
            app.NonfiberClassificationDataOutputFolder.FontName = 'Avenir';
            app.NonfiberClassificationDataOutputFolder.Position = [142 155 100 22];

            % Create ClassifyNonfiberObjects
            app.ClassifyNonfiberObjects = uibutton(app.NonfiberClassificationControlPanel, 'push');
            app.ClassifyNonfiberObjects.ButtonPushedFcn = createCallbackFcn(app, @ClassifyNonfiberObjectsButtonPushed, true);
            app.ClassifyNonfiberObjects.FontName = 'Avenir';
            app.ClassifyNonfiberObjects.Position = [19 113 100 24];
            app.ClassifyNonfiberObjects.Text = 'Calculate';

            % Create WritetoExcelNonfiberClassification
            app.WritetoExcelNonfiberClassification = uibutton(app.NonfiberClassificationControlPanel, 'push');
            app.WritetoExcelNonfiberClassification.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelNonfiberClassificationButtonPushed, true);
            app.WritetoExcelNonfiberClassification.FontName = 'Avenir';
            app.WritetoExcelNonfiberClassification.Enable = 'off';
            app.WritetoExcelNonfiberClassification.Position = [140 112 100 24];
            app.WritetoExcelNonfiberClassification.Text = 'Write to Excel';

            % Create DoneNonfiberClassification
            app.DoneNonfiberClassification = uibutton(app.NonfiberClassificationControlPanel, 'push');
            app.DoneNonfiberClassification.ButtonPushedFcn = createCallbackFcn(app, @DoneNonfiberClassificationButtonPushed, true);
            app.DoneNonfiberClassification.FontName = 'Avenir';
            app.DoneNonfiberClassification.Position = [84 45 100 24];
            app.DoneNonfiberClassification.Text = 'Close';

            % Create ClassificationColorLabel
            app.ClassificationColorLabel = uilabel(app.NonfiberClassificationControlPanel);
            app.ClassificationColorLabel.HorizontalAlignment = 'right';
            app.ClassificationColorLabel.FontName = 'Avenir';
            app.ClassificationColorLabel.Tooltip = {'Based on the channel used for imaging.'};
            app.ClassificationColorLabel.Position = [2 188 108 22];
            app.ClassificationColorLabel.Text = 'Classification Color';

            % Create NonfiberClassificationColorDropDown
            app.NonfiberClassificationColorDropDown = uidropdown(app.NonfiberClassificationControlPanel);
            app.NonfiberClassificationColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.NonfiberClassificationColorDropDown.ItemsData = {'1', '2', '3'};
            app.NonfiberClassificationColorDropDown.ValueChangedFcn = createCallbackFcn(app, @NonfiberClassificationColorDropDownValueChanged, true);
            app.NonfiberClassificationColorDropDown.FontName = 'Avenir';
            app.NonfiberClassificationColorDropDown.Position = [118 188 100 22];
            app.NonfiberClassificationColorDropDown.Value = '1';

            % Create NonfiberClassificationFileWriteStatusLabel
            app.NonfiberClassificationFileWriteStatusLabel = uilabel(app.NonfiberClassificationControlPanel);
            app.NonfiberClassificationFileWriteStatusLabel.Position = [74 76 125 28];
            app.NonfiberClassificationFileWriteStatusLabel.Text = {'Nonfiber Classification'; ' File Write Status'};

            % Create NonfiberClassificationDescription
            app.NonfiberClassificationDescription = uilabel(app.NonfiberClassificationControlPanel);
            app.NonfiberClassificationDescription.FontWeight = 'bold';
            app.NonfiberClassificationDescription.Position = [12 374 213 42];
            app.NonfiberClassificationDescription.Text = {'Detect non-fibers with intensity of a '; 'certain color channel greater than a'; 'threshold value. '};

            % Create NonfiberClassificationDescription_2
            app.NonfiberClassificationDescription_2 = uilabel(app.NonfiberClassificationControlPanel);
            app.NonfiberClassificationDescription_2.FontWeight = 'bold';
            app.NonfiberClassificationDescription_2.Position = [9 278 250 70];
            app.NonfiberClassificationDescription_2.Text = {'Set the field values below. Hover over the '; 'field names for more information. Press '; '"Calculate" to  calculate the nonfiber'; 'classification, and "Write to Excel" to save'; 'the data to Excel.'};

            % Create NonfiberPanel
            app.NonfiberPanel = uipanel(app.UIFigure);
            app.NonfiberPanel.Visible = 'off';
            app.NonfiberPanel.BackgroundColor = [1 1 1];
            app.NonfiberPanel.FontName = 'Avenir';
            app.NonfiberPanel.Position = [293 53 876 605];

            % Create NonfiberAxesR
            app.NonfiberAxesR = uiaxes(app.NonfiberPanel);
            xlabel(app.NonfiberAxesR, 'X')
            ylabel(app.NonfiberAxesR, 'Y')
            app.NonfiberAxesR.PlotBoxAspectRatio = [1.35976789168279 1 1];
            app.NonfiberAxesR.FontName = 'Avenir';
            app.NonfiberAxesR.XColor = 'none';
            app.NonfiberAxesR.YColor = 'none';
            app.NonfiberAxesR.Position = [442 92 401 427];

            % Create NonfiberAxesL
            app.NonfiberAxesL = uiaxes(app.NonfiberPanel);
            xlabel(app.NonfiberAxesL, 'X')
            ylabel(app.NonfiberAxesL, 'Y')
            app.NonfiberAxesL.PlotBoxAspectRatio = [1.35976789168279 1 1];
            app.NonfiberAxesL.FontName = 'Avenir';
            app.NonfiberAxesL.XColor = 'none';
            app.NonfiberAxesL.YColor = 'none';
            app.NonfiberAxesL.Position = [28 92 401 427];

            % Create ThresholdEditField_2Label_2
            app.ThresholdEditField_2Label_2 = uilabel(app.NonfiberPanel);
            app.ThresholdEditField_2Label_2.HorizontalAlignment = 'right';
            app.ThresholdEditField_2Label_2.FontName = 'Avenir';
            app.ThresholdEditField_2Label_2.Position = [204 10 59 22];
            app.ThresholdEditField_2Label_2.Text = 'Threshold';

            % Create NonfiberThreshold
            app.NonfiberThreshold = uieditfield(app.NonfiberPanel, 'numeric');
            app.NonfiberThreshold.Limits = [0 Inf];
            app.NonfiberThreshold.FontName = 'Avenir';
            app.NonfiberThreshold.Position = [278 10 100 22];

            % Create NonfiberAdjust
            app.NonfiberAdjust = uibutton(app.NonfiberPanel, 'push');
            app.NonfiberAdjust.ButtonPushedFcn = createCallbackFcn(app, @NonfiberAdjustButtonPushed, true);
            app.NonfiberAdjust.FontName = 'Avenir';
            app.NonfiberAdjust.Position = [418 9 100 24];
            app.NonfiberAdjust.Text = 'Adjust';

            % Create NonfiberAccept
            app.NonfiberAccept = uibutton(app.NonfiberPanel, 'push');
            app.NonfiberAccept.ButtonPushedFcn = createCallbackFcn(app, @NonfiberAcceptButtonPushed, true);
            app.NonfiberAccept.FontName = 'Avenir';
            app.NonfiberAccept.Position = [542 8 100 24];
            app.NonfiberAccept.Text = 'Accept';

            % Create NonfiberThresholdLabel
            app.NonfiberThresholdLabel = uilabel(app.NonfiberPanel);
            app.NonfiberThresholdLabel.FontWeight = 'bold';
            app.NonfiberThresholdLabel.Position = [160 37 646 42];
            app.NonfiberThresholdLabel.Text = {'The value displayed below is the recommended threshold value. To calculate nonfiber objects '; 'with a different threshold value, select "Adjust". Otherwise, select "Accept".'};

            % Create CentralNucleiPanel
            app.CentralNucleiPanel = uipanel(app.UIFigure);
            app.CentralNucleiPanel.Visible = 'off';
            app.CentralNucleiPanel.FontName = 'Avenir';
            app.CentralNucleiPanel.Position = [285 28 882 634];

            % Create CentralNucleiAxesR
            app.CentralNucleiAxesR = uiaxes(app.CentralNucleiPanel);
            app.CentralNucleiAxesR.PlotBoxAspectRatio = [1.34971644612476 1 1];
            app.CentralNucleiAxesR.FontName = 'Avenir';
            app.CentralNucleiAxesR.XColor = 'none';
            app.CentralNucleiAxesR.YColor = 'none';
            app.CentralNucleiAxesR.Position = [428 171 448 432];

            % Create CentralNucleiAxesL
            app.CentralNucleiAxesL = uiaxes(app.CentralNucleiPanel);
            app.CentralNucleiAxesL.PlotBoxAspectRatio = [1.34971644612476 1 1];
            app.CentralNucleiAxesL.FontName = 'Avenir';
            app.CentralNucleiAxesL.XColor = 'none';
            app.CentralNucleiAxesL.YColor = 'none';
            app.CentralNucleiAxesL.Position = [1 171 448 432];

            % Create ThresholdEditField_2Label
            app.ThresholdEditField_2Label = uilabel(app.CentralNucleiPanel);
            app.ThresholdEditField_2Label.HorizontalAlignment = 'right';
            app.ThresholdEditField_2Label.FontName = 'Avenir';
            app.ThresholdEditField_2Label.Position = [206 38 59 22];
            app.ThresholdEditField_2Label.Text = 'Threshold';

            % Create ThresholdCentralNuclei
            app.ThresholdCentralNuclei = uieditfield(app.CentralNucleiPanel, 'numeric');
            app.ThresholdCentralNuclei.Limits = [0 Inf];
            app.ThresholdCentralNuclei.FontName = 'Avenir';
            app.ThresholdCentralNuclei.Position = [280 38 100 22];

            % Create AdjustCentralNuclei
            app.AdjustCentralNuclei = uibutton(app.CentralNucleiPanel, 'push');
            app.AdjustCentralNuclei.ButtonPushedFcn = createCallbackFcn(app, @AdjustCentralNucleiButtonPushed, true);
            app.AdjustCentralNuclei.FontName = 'Avenir';
            app.AdjustCentralNuclei.Position = [424 37 100 24];
            app.AdjustCentralNuclei.Text = 'Adjust';

            % Create AcceptCentralNuclei
            app.AcceptCentralNuclei = uibutton(app.CentralNucleiPanel, 'push');
            app.AcceptCentralNuclei.ButtonPushedFcn = createCallbackFcn(app, @AcceptCentralNucleiButtonPushed, true);
            app.AcceptCentralNuclei.FontName = 'Avenir';
            app.AcceptCentralNuclei.Position = [555 37 100 24];
            app.AcceptCentralNuclei.Text = 'Accept';

            % Create Label
            app.Label = uilabel(app.CentralNucleiPanel);
            app.Label.FontWeight = 'bold';
            app.Label.Position = [206 76 536 28];
            app.Label.Text = {'The value displayed below is the recommended threshold value. To calculate central nuclei'; 'with a different threshold value, select "Adjust". Otherwise, select "Accept".'};

            % Create FiberTypingPanel
            app.FiberTypingPanel = uipanel(app.UIFigure);
            app.FiberTypingPanel.Visible = 'off';
            app.FiberTypingPanel.FontName = 'Avenir';
            app.FiberTypingPanel.Position = [285 15 882 650];

            % Create FiberTypingAxesL
            app.FiberTypingAxesL = uiaxes(app.FiberTypingPanel);
            app.FiberTypingAxesL.PlotBoxAspectRatio = [1.06832298136646 1 1];
            app.FiberTypingAxesL.FontName = 'Avenir';
            app.FiberTypingAxesL.XColor = 'none';
            app.FiberTypingAxesL.YColor = 'none';
            app.FiberTypingAxesL.Position = [22 289 414 332];

            % Create FiberTypingAxesR
            app.FiberTypingAxesR = uiaxes(app.FiberTypingPanel);
            app.FiberTypingAxesR.PlotBoxAspectRatio = [1.06832298136646 1 1];
            app.FiberTypingAxesR.FontName = 'Avenir';
            app.FiberTypingAxesR.XColor = 'none';
            app.FiberTypingAxesR.YColor = 'none';
            app.FiberTypingAxesR.Position = [450 289 416 335];

            % Create FThistL
            app.FThistL = uiaxes(app.FiberTypingPanel);
            app.FThistL.PlotBoxAspectRatio = [1.83030303030303 1 1];
            app.FThistL.FontName = 'Avenir';
            app.FThistL.Position = [22 83 402 207];

            % Create FThistR
            app.FThistR = uiaxes(app.FiberTypingPanel);
            app.FThistR.PlotBoxAspectRatio = [1.91358024691358 1 1];
            app.FThistR.FontName = 'Avenir';
            app.FThistR.Position = [450 86 402 204];

            % Create ThresholdEditFieldLabel
            app.ThresholdEditFieldLabel = uilabel(app.FiberTypingPanel);
            app.ThresholdEditFieldLabel.HorizontalAlignment = 'right';
            app.ThresholdEditFieldLabel.FontName = 'Avenir';
            app.ThresholdEditFieldLabel.Enable = 'off';
            app.ThresholdEditFieldLabel.Position = [210 15 59 22];
            app.ThresholdEditFieldLabel.Text = 'Threshold';

            % Create ThresholdEditField
            app.ThresholdEditField = uieditfield(app.FiberTypingPanel, 'numeric');
            app.ThresholdEditField.Limits = [0 Inf];
            app.ThresholdEditField.FontName = 'Avenir';
            app.ThresholdEditField.Enable = 'off';
            app.ThresholdEditField.Position = [284 14 100 22];

            % Create AdjustButton
            app.AdjustButton = uibutton(app.FiberTypingPanel, 'push');
            app.AdjustButton.ButtonPushedFcn = createCallbackFcn(app, @AdjustButtonPushed, true);
            app.AdjustButton.FontName = 'Avenir';
            app.AdjustButton.Enable = 'off';
            app.AdjustButton.Position = [423 12 100 24];
            app.AdjustButton.Text = 'Adjust';

            % Create AcceptButton
            app.AcceptButton = uibutton(app.FiberTypingPanel, 'push');
            app.AcceptButton.ButtonPushedFcn = createCallbackFcn(app, @AcceptButtonPushed, true);
            app.AcceptButton.FontName = 'Avenir';
            app.AcceptButton.Enable = 'off';
            app.AcceptButton.Position = [548 12 100 24];
            app.AcceptButton.Text = 'Accept';

            % Create Label_2
            app.Label_2 = uilabel(app.FiberTypingPanel);
            app.Label_2.FontWeight = 'bold';
            app.Label_2.Position = [176 50 523 28];
            app.Label_2.Text = {'The value displayed below is the recommended threshold value. To calculate fiber typing'; 'with a different threshold value, select "Adjust". Otherwise, select "Accept".'};

            % Create NonfiberClassificationPanel
            app.NonfiberClassificationPanel = uipanel(app.UIFigure);
            app.NonfiberClassificationPanel.BorderType = 'none';
            app.NonfiberClassificationPanel.Visible = 'off';
            app.NonfiberClassificationPanel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.NonfiberClassificationPanel.FontName = 'Avenir';
            app.NonfiberClassificationPanel.Position = [282 52 876 605];

            % Create NonfiberClassificationAxes_R
            app.NonfiberClassificationAxes_R = uiaxes(app.NonfiberClassificationPanel);
            xlabel(app.NonfiberClassificationAxes_R, 'X')
            ylabel(app.NonfiberClassificationAxes_R, 'Y')
            app.NonfiberClassificationAxes_R.PlotBoxAspectRatio = [1.35976789168279 1 1];
            app.NonfiberClassificationAxes_R.FontName = 'Avenir';
            app.NonfiberClassificationAxes_R.XColor = 'none';
            app.NonfiberClassificationAxes_R.YColor = 'none';
            app.NonfiberClassificationAxes_R.Position = [450 90 405 469];

            % Create NonfiberClassificationAxes_L
            app.NonfiberClassificationAxes_L = uiaxes(app.NonfiberClassificationPanel);
            xlabel(app.NonfiberClassificationAxes_L, 'X')
            ylabel(app.NonfiberClassificationAxes_L, 'Y')
            app.NonfiberClassificationAxes_L.PlotBoxAspectRatio = [1.35976789168279 1 1];
            app.NonfiberClassificationAxes_L.FontName = 'Avenir';
            app.NonfiberClassificationAxes_L.XColor = 'none';
            app.NonfiberClassificationAxes_L.YColor = 'none';
            app.NonfiberClassificationAxes_L.Position = [2 90 405 469];

            % Create ThresholdEditField_2Label_3
            app.ThresholdEditField_2Label_3 = uilabel(app.NonfiberClassificationPanel);
            app.ThresholdEditField_2Label_3.HorizontalAlignment = 'right';
            app.ThresholdEditField_2Label_3.FontName = 'Avenir';
            app.ThresholdEditField_2Label_3.Position = [215 16 59 22];
            app.ThresholdEditField_2Label_3.Text = 'Threshold';

            % Create NonfiberClassificationThreshold
            app.NonfiberClassificationThreshold = uieditfield(app.NonfiberClassificationPanel, 'numeric');
            app.NonfiberClassificationThreshold.Limits = [0 Inf];
            app.NonfiberClassificationThreshold.Editable = 'off';
            app.NonfiberClassificationThreshold.FontName = 'Avenir';
            app.NonfiberClassificationThreshold.Enable = 'off';
            app.NonfiberClassificationThreshold.Position = [289 16 100 22];

            % Create NonfiberClassificationAdjust
            app.NonfiberClassificationAdjust = uibutton(app.NonfiberClassificationPanel, 'push');
            app.NonfiberClassificationAdjust.ButtonPushedFcn = createCallbackFcn(app, @NonfiberClassificationAdjustButtonPushed, true);
            app.NonfiberClassificationAdjust.FontName = 'Avenir';
            app.NonfiberClassificationAdjust.Enable = 'off';
            app.NonfiberClassificationAdjust.Position = [429 15 100 24];
            app.NonfiberClassificationAdjust.Text = 'Adjust';

            % Create NonfiberClassificationAccept
            app.NonfiberClassificationAccept = uibutton(app.NonfiberClassificationPanel, 'push');
            app.NonfiberClassificationAccept.ButtonPushedFcn = createCallbackFcn(app, @NonfiberClassificationAcceptButtonPushed, true);
            app.NonfiberClassificationAccept.FontName = 'Avenir';
            app.NonfiberClassificationAccept.Enable = 'off';
            app.NonfiberClassificationAccept.Position = [553 14 100 24];
            app.NonfiberClassificationAccept.Text = 'Accept';

            % Create PercentPositiveLabel
            app.PercentPositiveLabel = uilabel(app.NonfiberClassificationPanel);
            app.PercentPositiveLabel.HorizontalAlignment = 'right';
            app.PercentPositiveLabel.FontName = 'Avenir';
            app.PercentPositiveLabel.Position = [323 147 94 22];
            app.PercentPositiveLabel.Text = 'Percent Positive:';

            % Create PercentPositiveTextArea
            app.PercentPositiveTextArea = uitextarea(app.NonfiberClassificationPanel);
            app.PercentPositiveTextArea.Editable = 'off';
            app.PercentPositiveTextArea.FontName = 'Avenir';
            app.PercentPositiveTextArea.Position = [431 147 150 24];

            % Create OriginalImageLabel
            app.OriginalImageLabel = uilabel(app.NonfiberClassificationPanel);
            app.OriginalImageLabel.FontName = 'Avenir';
            app.OriginalImageLabel.FontWeight = 'bold';
            app.OriginalImageLabel.Position = [177 458 89 22];
            app.OriginalImageLabel.Text = 'Original Image';

            % Create PositiveNonfiberObjectsLabel
            app.PositiveNonfiberObjectsLabel = uilabel(app.NonfiberClassificationPanel);
            app.PositiveNonfiberObjectsLabel.FontName = 'Avenir';
            app.PositiveNonfiberObjectsLabel.FontWeight = 'bold';
            app.PositiveNonfiberObjectsLabel.Position = [589 462 150 22];
            app.PositiveNonfiberObjectsLabel.Text = 'Positive Nonfiber Objects';

            % Create Label_3
            app.Label_3 = uilabel(app.NonfiberClassificationPanel);
            app.Label_3.FontWeight = 'bold';
            app.Label_3.Position = [153 53 626 28];
            app.Label_3.Text = {'The value displayed below is the recommended threshold value. To calculate nonfiber object classification'; 'with a different threshold value, select "Adjust". Otherwise, select "Accept".'};

            % Create PropertiesPanel
            app.PropertiesPanel = uipanel(app.UIFigure);
            app.PropertiesPanel.Visible = 'off';
            app.PropertiesPanel.FontName = 'Avenir';
            app.PropertiesPanel.Position = [285 28 890 637];

            % Create FeretAxes
            app.FeretAxes = uiaxes(app.PropertiesPanel);
            title(app.FeretAxes, 'Minimum Feret Diameter (um)')
            app.FeretAxes.PlotBoxAspectRatio = [3.2695652173913 1 1];
            app.FeretAxes.FontName = 'Avenir';
            app.FeretAxes.Position = [53 323 799 285];

            % Create FiberSizeAxes
            app.FiberSizeAxes = uiaxes(app.PropertiesPanel);
            title(app.FiberSizeAxes, 'Fiber Area (um^2)')
            app.FiberSizeAxes.PlotBoxAspectRatio = [3.29824561403509 1 1];
            app.FiberSizeAxes.FontName = 'Avenir';
            app.FiberSizeAxes.Position = [53 25 799 285];

            % Create FiberTypingControlPanel
            app.FiberTypingControlPanel = uipanel(app.UIFigure);
            app.FiberTypingControlPanel.Visible = 'off';
            app.FiberTypingControlPanel.BackgroundColor = [1 1 1];
            app.FiberTypingControlPanel.FontName = 'Avenir';
            app.FiberTypingControlPanel.Position = [10 176 260 385];

            % Create FiberTypingChannelColorBox
            app.FiberTypingChannelColorBox = uiaxes(app.FiberTypingControlPanel);
            app.FiberTypingChannelColorBox.Toolbar.Visible = 'off';
            app.FiberTypingChannelColorBox.FontName = 'Avenir';
            app.FiberTypingChannelColorBox.XTick = [];
            app.FiberTypingChannelColorBox.YTick = [];
            app.FiberTypingChannelColorBox.Color = [0 1 1];
            app.FiberTypingChannelColorBox.Box = 'on';
            app.FiberTypingChannelColorBox.Position = [229 181 30 30];

            % Create PixelSizeumpixelEditField_3Label
            app.PixelSizeumpixelEditField_3Label = uilabel(app.FiberTypingControlPanel);
            app.PixelSizeumpixelEditField_3Label.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditField_3Label.FontName = 'Avenir';
            app.PixelSizeumpixelEditField_3Label.Tooltip = {'Pixel size is default based on microscope, adjust pixel size to your specific microscope.'};
            app.PixelSizeumpixelEditField_3Label.Position = [44 212 59 32];
            app.PixelSizeumpixelEditField_3Label.Text = {'Pixel Size'; '(um/pixel)'};

            % Create PixelSizeFiberTyping
            app.PixelSizeFiberTyping = uieditfield(app.FiberTypingControlPanel, 'numeric');
            app.PixelSizeFiberTyping.Limits = [0 Inf];
            app.PixelSizeFiberTyping.ValueChangedFcn = createCallbackFcn(app, @PixelSizeFiberTypingValueChanged, true);
            app.PixelSizeFiberTyping.FontName = 'Avenir';
            app.PixelSizeFiberTyping.Position = [118 222 100 22];

            % Create DataOutputFolderEditField_3Label
            app.DataOutputFolderEditField_3Label = uilabel(app.FiberTypingControlPanel);
            app.DataOutputFolderEditField_3Label.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_3Label.FontName = 'Avenir';
            app.DataOutputFolderEditField_3Label.FontWeight = 'bold';
            app.DataOutputFolderEditField_3Label.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.DataOutputFolderEditField_3Label.Position = [13 149 114 22];
            app.DataOutputFolderEditField_3Label.Text = 'Data Output Folder';

            % Create FiberTypingDataOutputFolder
            app.FiberTypingDataOutputFolder = uieditfield(app.FiberTypingControlPanel, 'text');
            app.FiberTypingDataOutputFolder.FontName = 'Avenir';
            app.FiberTypingDataOutputFolder.Tooltip = {'Save Excel sheet of this step to the folder specified here.'};
            app.FiberTypingDataOutputFolder.Position = [142 149 100 22];

            % Create FiberTypeColorDropDownLabel
            app.FiberTypeColorDropDownLabel = uilabel(app.FiberTypingControlPanel);
            app.FiberTypeColorDropDownLabel.HorizontalAlignment = 'right';
            app.FiberTypeColorDropDownLabel.FontName = 'Avenir';
            app.FiberTypeColorDropDownLabel.Tooltip = {'Based on the channel used for imaging.'};
            app.FiberTypeColorDropDownLabel.Position = [16 184 95 22];
            app.FiberTypeColorDropDownLabel.Text = 'Fiber Type Color';

            % Create FiberTypeColorDropDown
            app.FiberTypeColorDropDown = uidropdown(app.FiberTypingControlPanel);
            app.FiberTypeColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.FiberTypeColorDropDown.ItemsData = {'1', '2', '3'};
            app.FiberTypeColorDropDown.ValueChangedFcn = createCallbackFcn(app, @FiberTypeColorValueChanged, true);
            app.FiberTypeColorDropDown.FontName = 'Avenir';
            app.FiberTypeColorDropDown.Position = [126 184 100 22];
            app.FiberTypeColorDropDown.Value = '1';

            % Create CalculateFiberTyping
            app.CalculateFiberTyping = uibutton(app.FiberTypingControlPanel, 'push');
            app.CalculateFiberTyping.ButtonPushedFcn = createCallbackFcn(app, @CalculateFiberTypingButtonPushed, true);
            app.CalculateFiberTyping.FontName = 'Avenir';
            app.CalculateFiberTyping.Position = [30 96 100 24];
            app.CalculateFiberTyping.Text = 'Calculate';

            % Create WritetoExcelFT
            app.WritetoExcelFT = uibutton(app.FiberTypingControlPanel, 'push');
            app.WritetoExcelFT.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelFTButtonPushed, true);
            app.WritetoExcelFT.FontName = 'Avenir';
            app.WritetoExcelFT.Enable = 'off';
            app.WritetoExcelFT.Position = [138 96 100 24];
            app.WritetoExcelFT.Text = 'Write to Excel';

            % Create DoneFiberTyping
            app.DoneFiberTyping = uibutton(app.FiberTypingControlPanel, 'push');
            app.DoneFiberTyping.ButtonPushedFcn = createCallbackFcn(app, @DoneFiberTypingButtonPushed, true);
            app.DoneFiberTyping.FontName = 'Avenir';
            app.DoneFiberTyping.Position = [87 26 100 24];
            app.DoneFiberTyping.Text = 'Close';

            % Create FiberTypingDescription
            app.FiberTypingDescription = uilabel(app.FiberTypingControlPanel);
            app.FiberTypingDescription.FontWeight = 'bold';
            app.FiberTypingDescription.Position = [9 321 231 42];
            app.FiberTypingDescription.Text = {'Detect fibers with intensity of a certain '; 'color channel greater than a threshold '; 'value. '};

            % Create FiberTypingDescription_2
            app.FiberTypingDescription_2 = uilabel(app.FiberTypingControlPanel);
            app.FiberTypingDescription_2.FontWeight = 'bold';
            app.FiberTypingDescription_2.Position = [9 252 241 56];
            app.FiberTypingDescription_2.Text = {'Set the field values below. Hover over '; 'the field names for more information. '; 'Press "Calculate" and "Write to Excel" to'; 'save the data to Excel.'; ''};

            % Create FiberTypingFileWriteStatusLabel
            app.FiberTypingFileWriteStatusLabel = uilabel(app.FiberTypingControlPanel);
            app.FiberTypingFileWriteStatusLabel.Position = [34 65 162 22];
            app.FiberTypingFileWriteStatusLabel.Text = 'Fiber Typing File Write Status';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = SMASH_ML_1_2_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end