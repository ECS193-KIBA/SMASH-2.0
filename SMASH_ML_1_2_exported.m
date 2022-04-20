classdef SMASH_ML_1_2_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        ManualFilterControls            matlab.ui.container.Panel
        MergeObjectsButton              matlab.ui.control.Button
        FinishManualFilteringButton     matlab.ui.control.Button
        RemoveObjectsButton             matlab.ui.control.Button
        Toolbar                         matlab.ui.container.Panel
        NonfiberObjectsButton           matlab.ui.control.Button
        FiberTypingButton               matlab.ui.control.Button
        CentralNucleiButton             matlab.ui.control.Button
        FiberPropertiesButton           matlab.ui.control.Button
        ManualFiberFilterButton         matlab.ui.control.Button
        FiberPredictionButton           matlab.ui.control.Button
        ManualSegmentationButton        matlab.ui.control.Button
        InitialSegmentationButton       matlab.ui.control.Button
        NonfiberControlPanel            matlab.ui.container.Panel
        PixelSizeNonfiber               matlab.ui.control.NumericEditField
        PixelSizeumpixelLabel_2         matlab.ui.control.Label
        DoneNonfiber                    matlab.ui.control.Button
        WritetoExcelNonfiber            matlab.ui.control.Button
        CalculateNonfiberObjects        matlab.ui.control.Button
        NonfiberObjectsDataOutputFolder  matlab.ui.control.EditField
        DataOutputFolderEditField_3Label_2  matlab.ui.control.Label
        NonfiberObjectColor             matlab.ui.control.DropDown
        ObjectColorDropDownLabel        matlab.ui.control.Label
        NonfiberPanel                   matlab.ui.container.Panel
        NonfiberAccept                  matlab.ui.control.Button
        NonfiberAdjust                  matlab.ui.control.Button
        NonfiberThreshold               matlab.ui.control.NumericEditField
        ThresholdEditField_2Label_2     matlab.ui.control.Label
        NonfiberAxes                    matlab.ui.control.UIAxes
        CNFPanel                        matlab.ui.container.Panel
        AcceptCNF                       matlab.ui.control.Button
        AdjustCNF                       matlab.ui.control.Button
        ThresholdCNF                    matlab.ui.control.NumericEditField
        ThresholdEditField_2Label       matlab.ui.control.Label
        CNFAxes                         matlab.ui.control.UIAxes
        FiberTypingPanel                matlab.ui.container.Panel
        AcceptButton                    matlab.ui.control.Button
        AdjustButton                    matlab.ui.control.Button
        ThresholdEditField              matlab.ui.control.NumericEditField
        ThresholdEditFieldLabel         matlab.ui.control.Label
        FThistR                         matlab.ui.control.UIAxes
        FThistL                         matlab.ui.control.UIAxes
        FTAxesR                         matlab.ui.control.UIAxes
        FTAxesL                         matlab.ui.control.UIAxes
        FiberTypingControlPanel         matlab.ui.container.Panel
        DoneFT                          matlab.ui.control.Button
        WritetoExcelFT                  matlab.ui.control.Button
        CalculateFiberTyping            matlab.ui.control.Button
        FiberTypeColorDropDown          matlab.ui.control.DropDown
        FiberTypeColorDropDownLabel     matlab.ui.control.Label
        FiberTypingDataOutputFolder     matlab.ui.control.EditField
        DataOutputFolderEditField_3Label  matlab.ui.control.Label
        PixelSizeFiberType              matlab.ui.control.NumericEditField
        PixelSizeumpixelEditField_3Label  matlab.ui.control.Label
        CNFControlPanel                 matlab.ui.container.Panel
        CentralNucleiDataOutputFolder   matlab.ui.control.EditField
        DataOutputFolderEditField_2Label  matlab.ui.control.Label
        MinimumNucleusSizeum2EditField  matlab.ui.control.NumericEditField
        MinimumNucleusSizeum2EditFieldLabel  matlab.ui.control.Label
        DistancefromborderEditField     matlab.ui.control.NumericEditField
        DistancefromborderEditFieldLabel  matlab.ui.control.Label
        DoneButton_CNF                  matlab.ui.control.Button
        CNFExcelWrite                   matlab.ui.control.Button
        CalculateCentralNuclei          matlab.ui.control.Button
        NucleiColorDropDown             matlab.ui.control.DropDown
        NucleiColorDropDownLabel        matlab.ui.control.Label
        PixelSizeumpixelEditField_2     matlab.ui.control.NumericEditField
        PixelSizeumpixelLabel           matlab.ui.control.Label
        FiberPredictionControlPanel     matlab.ui.container.Panel
        SortingThresholdSlider          matlab.ui.control.Slider
        SortingThresholdHigherrequiresmoremanualsortingLabel  matlab.ui.control.Label
        ManualSortingButton             matlab.ui.control.Button
        FilterButton                    matlab.ui.control.Button
        PropertiesControlPanel          matlab.ui.container.Panel
        CalculateFiberProperties        matlab.ui.control.Button
        PixelSizeumpixelEditField       matlab.ui.control.NumericEditField
        PixelSizeumpixelEditFieldLabel_2  matlab.ui.control.Label
        FiberPropertiesDataOutputFolder  matlab.ui.control.EditField
        DataOutputFolderEditFieldLabel  matlab.ui.control.Label
        DoneButton                      matlab.ui.control.Button
        WritetoExcelButton              matlab.ui.control.Button
        PropertiesPanel                 matlab.ui.container.Panel
        FiberSizeAxes                   matlab.ui.control.UIAxes
        FeretAxes                       matlab.ui.control.UIAxes
        SortingAxesPanel                matlab.ui.container.Panel
        MarkasfiberLabel                matlab.ui.control.Label
        NoButton                        matlab.ui.control.Button
        YesButton                       matlab.ui.control.Button
        UIAxesR                         matlab.ui.control.UIAxes
        UIAxesL                         matlab.ui.control.UIAxes
        DrawLineControls                matlab.ui.container.Panel
        FinishSegmentingButton          matlab.ui.control.Button
        DoneDrawingButton               matlab.ui.control.Button
        DrawLineButton                  matlab.ui.control.Button
        Image2                          matlab.ui.control.Image
        SMASHLabel                      matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        Prompt                          matlab.ui.control.Label
        SegmentationParameters          matlab.ui.container.Panel
        DetectValueButton               matlab.ui.control.Button
        SegmentationThresholdSlider     matlab.ui.control.Slider
        SegmentationThresholdSliderLabel  matlab.ui.control.Label
        AcceptSegmentationButton        matlab.ui.control.Button
        FiberOutlineColorDropDown       matlab.ui.control.DropDown
        FiberOutlineColorDropDownLabel  matlab.ui.control.Label
        SegmentButton                   matlab.ui.control.Button
        PixelSizeField                  matlab.ui.control.NumericEditField
        PixelSizeumpixelEditFieldLabel  matlab.ui.control.Label
        FilenameLabel                   matlab.ui.control.Label
        SelectFileButton                matlab.ui.control.Button
        UIAxes                          matlab.ui.control.UIAxes
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
        CNF_Adj
        Obj_Adj
        thresh_nf
        num_nf
        nf_data
        segmodel
    end
    
    methods (Access = private)

        function results = CreateFolderIfDirectoryIsNonexistent(app, pathDirectory)
            if exist(pathDirectory,'dir') == 0
                mkdir(pathDirectory)
            end
        end
        
        function results = GetBoldBoundary(app, label, region_id)
            bw = CreateEmptyImageBWImageWithAppOrigImgDimensions(app);
            bw(label == region_id) = 1;

            left = ShiftLeft(app, bw);
            right = ShiftRight(app, bw);
            up = ShiftUp(app, bw);
            down = ShiftDown(app, bw);
            upleft = ShiftUp(app, left);
            upright = ShiftUp(app, right);
            downleft = ShiftDown(app, left);
            downright = ShiftDown(app, right);
            results = bw | up | down | left | right | upleft | upright | downleft | downright;
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
        
        function results = IsPointAdjacentToAcceptableLabels(app, r, c, object_labels, first_region_to_merge, second_region_to_merge)
            point_neighbor_labels = [object_labels(r+1,c), object_labels(r-1,c), object_labels(r,c+1), object_labels(r,c-1)];
            % A neighbor is acceptable as long as it's either a non-object
            % (0) or one of two objects being merge.
            acceptable_labels = [0 first_region_to_merge second_region_to_merge];
            results = all(ismember(point_neighbor_labels, acceptable_labels));
        end
        
        function results = ShiftLeft(app, mat)
            results = circshift(mat,[0 -1]);
            results(:,end) = 0;
        end

        function results = ShiftRight(app, mat)
            results = circshift(mat,[0 1]);
            results(:,1) = 0;
        end

        function results = ShiftUp(app, mat)
            results = circshift(mat,[-1 0]);
            results(end,:) = 0;
        end

        function results = ShiftDown(app, mat)
            results = circshift(mat,[1 0]);
            results(1,:) = 0;
        end
        
        function results = MergeObjects(app, label, first_region_to_merge, second_region_to_merge)
            bw_first_region_bolded_boundary = GetBoldBoundary(app, label, first_region_to_merge);
            bw_second_region_bolded_boundary = GetBoldBoundary(app, label, second_region_to_merge);
            bw_boundary_intersection = bw_first_region_bolded_boundary & bw_second_region_bolded_boundary;
            bw_pixels_required_to_merge = FilterPointsThatWouldCauseUndesiredMerge(app, bw_boundary_intersection, label, first_region_to_merge, second_region_to_merge);
            
            results = app.bw_obj;
            results(bw_pixels_required_to_merge == 1) = 1;
        end
        
        function results = CreateEmptyImageBWImageWithAppOrigImgDimensions(app)
             % [1 2] specifies that we only want the width and height
             % dimension respectively, and ignore the 3rd dimension, which
             % 3 for RGB channels.
            app_orig_img_dimensions = size(app.orig_img, [1 2]);
            results = zeros(app_orig_img_dimensions, "logical");
        end
        
        function results = IsROIPositionInBound(app, xp, yp)
            xp_is_valid = xp > 0 & xp <= size(app.bw_obj,2);
            yp_is_valid = yp > 0 & yp <= size(app.bw_obj,1);
            results = xp_is_valid && yp_is_valid;
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
            app.PixelSizeField.Value = app.default{2,2};
            app.PixelSizeumpixelEditField.Value = app.default{2,2};
            %app.SegmentationThresholdSlider.Value = app.default{7,2};
            app.FiberOutlineColorDropDown.Value = num2str(app.default{3,2});
            app.NucleiColorDropDown.Value = num2str(app.default{4,2});
            app.PixelSizeumpixelEditField_2.Value = app.default{2,2};
            app.DistancefromborderEditField.Value = app.default{14,2};
            app.MinimumNucleusSizeum2EditField.Value = app.default{15,2};
            app.PixelSizeFiberType.Value = app.default{2,2};
            app.FiberTypeColorDropDown.Value = num2str(app.default{5,2});
            app.PixelSizeNonfiber.Value = app.default{2,2};
            app.NonfiberObjectColor.Value = num2str(app.default{6,2});
        end

        % Button pushed function: SelectFileButton
        function SelectFileButtonPushed(app, event)
            [FileName,PathName,FilterIndex] = uigetfile({'*.tif';'*.tiff';'*.jpg';'*.png';'*.bmp';'*.czi'},'File Selector - dont select mask');
            drawnow limitrate;
            figure(app.UIFigure)
            if FilterIndex
                C = strsplit(FileName,'.');
                ExtName = C(end);
                FileNameS = (C(1:(end-1)));
                FileNameS = FileNameS{1};
                MaskName = strcat(FileNameS,'_mask.',ExtName);
                MaskName = MaskName{1};
            end
            
            app.FilenameLabel.Text = FileNameS;
            app.Files{1} = FileName;
            app.Files{2} = MaskName;
            app.Files{3} = PathName;
            app.Files{4} = FileNameS;
            cd(PathName)
    
            % Change output directory to where image is located
            app.FiberPropertiesDataOutputFolder.Value = pwd;
            app.CentralNucleiDataOutputFolder.Value = pwd;
            app.FiberTypingDataOutputFolder.Value = pwd;
            app.NonfiberObjectsDataOutputFolder.Value = pwd;
            
            BioformatsData = bfopen(FileName);
            PixelDataForAllLayers = BioformatsData{1,1};
            ColorMapDataForAllLayers = BioformatsData{1, 3};
            isMultilayerImage = ~isempty(ColorMapDataForAllLayers{1,1});
            if isMultilayerImage
                LayerOnePixelData = PixelDataForAllLayers{1,1};
                LayerSize = size(LayerOnePixelData);
                RGBSize = [LayerSize 3];

                TotalRGB = zeros(RGBSize, 'uint8');
                TotalMultiSpectral = [];
                TotalColorDropDownItems = {};
                TotalColorDropDownItemsData = {};

                NumLayers = length(PixelDataForAllLayers);
                for Layer = 1:NumLayers
                    PixelsGrayscale = PixelDataForAllLayers{Layer, 1};
                    ColorMap = ColorMapDataForAllLayers{1, Layer};
                    PixelsRGBAsDouble = ind2rgb(PixelsGrayscale, ColorMap);
                    PixelsRGBAsUInt8 = im2uint8(PixelsRGBAsDouble);
                    PixelsGrayscaleUInt8 = im2uint8(PixelsGrayscale);

                    TotalMultiSpectral = cat(3, TotalMultiSpectral, PixelsGrayscaleUInt8);
                    TotalRGB = imadd(TotalRGB, PixelsRGBAsUInt8);
                    TotalColorDropDownItems = cat(2, TotalColorDropDownItems, {['Channel ', num2str(Layer)]});
                    TotalColorDropDownItemsData  = cat(2, TotalColorDropDownItemsData, {num2str(Layer)});
                end
                app.FiberOutlineColorDropDown.Items = TotalColorDropDownItems;
                app.FiberOutlineColorDropDown.ItemsData = TotalColorDropDownItemsData;
                app.NucleiColorDropDown.Items = TotalColorDropDownItems;
                app.NucleiColorDropDown.ItemsData = TotalColorDropDownItemsData;
                app.FiberTypeColorDropDown.Items = TotalColorDropDownItems;
                app.FiberTypeColorDropDown.ItemsData = TotalColorDropDownItemsData;
                app.NonfiberObjectColor.Items = TotalColorDropDownItems;
                app.NonfiberObjectColor.ItemsData = TotalColorDropDownItemsData;
                app.orig_img = TotalRGB;
                app.orig_img_multispectral = TotalMultiSpectral;
            else
                ImageData = imread(FileName);
                app.orig_img = ImageData;
                app.orig_img_multispectral = ImageData;
            end

            imshow(app.orig_img,'Parent',app.UIAxes);
            
            if exist(MaskName,'file')
                app.InitialSegmentationButton.Enable = 'on';
                app.FiberPredictionButton.Enable = 'on';
                app.ManualSegmentationButton.Enable = 'on';
                app.ManualFiberFilterButton.Enable = 'on';
                app.FiberPropertiesButton.Enable = 'on';
                app.CentralNucleiButton.Enable = 'on';
                app.FiberTypingButton.Enable = 'on';
                app.NonfiberObjectsButton.Enable = 'on';
            else
                app.InitialSegmentationButton.Enable = 'on';
            end
            
            
        end

        % Button pushed function: SegmentButton
        function SegmentButtonPushed(app, event)
            go = 1;
            app.pix_size = app.PixelSizeField.Value;
               files = dir;
               
               % Warn the user if a mask file already exists
               if find(strcmp({files.name},app.Files{2}),1) > 0
                   warn = uiconfirm(app.UIFigure,'Overwrite existing mask?','Confirm mask overwrite','Icon','Warning');
                   warn = convertCharsToStrings(warn);
                   if strcmp(warn,'Cancel')
                       go = 0;
                   end
               end
               
               if go
                   %orig_img = imread(app.Files{1});
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
                   app.AcceptSegmentationButton.Enable = 'on';
                   
                   
               end 
        end

        % Button pushed function: DrawLineButton
        function DrawLineButtonPushed(app, event)
            app.DoneDrawingButton.Enable = 'on';
            app.DrawLineButton.Enable = 'off';
            app.FinishSegmentingButton.Enable = 'off';
            app.Prompt.Text = 'Draw line to separate fibers and adjust as needed. Right click line to delete.';
            h = drawfreehand(app.UIAxes,'Closed',false,'FaceAlpha',0);
            uiwait(app.UIFigure);
            app.DoneDrawingButton.Enable = 'off';
            if isvalid(h)   % Checks if line exists or was deleted
                mask = createMask(h);
                mask2 = bwmorph(mask,'bridge');
                app.bw_obj = logical(app.bw_obj + mask2);
                flat_img = flattenMaskOverlay(app.orig_img, app.bw_obj, 1, 'w');
                imshow(flat_img,'Parent',app.UIAxes);
            end 
            app.DrawLineButton.Enable = 'on';
            app.FinishSegmentingButton.Enable = 'on';
        end

        % Button pushed function: DoneDrawingButton
        function DoneDrawingButtonPushed(app, event)
            uiresume(app.UIFigure);
            app.Prompt.Text = '';
        end

        % Button pushed function: FinishSegmentingButton
        function FinishSegmentingButtonPushed(app, event)
            label = bwlabel(~logical(app.bw_obj),4);
            rgb_label = label2rgb(label,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff');
            app.DrawLineControls.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            app.SegmentationParameters.Visible = 'off';
        end

        % Button pushed function: FilterButton
        function FilterButtonPushed(app, event)
            app.FilterButton.Enable = 'off';
            app.SortingThresholdSlider.Enable = 'off';
            app.Prompt.Text = 'Filtering, please wait.';
            
            drawnow limitrate
            app.pix_size = app.PixelSizeField.Value;
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
                    imshow(codedim,'Parent',app.UIAxesL);
                    hold(app.UIAxesL,'on');
                    plot(cents(maybe(i),1),cents(maybe(i),2),'y*','Parent',app.UIAxesL)
                    xlim(app.UIAxesL,[(cents(maybe(i),1)-100) (cents(maybe(i),1)+100)])
                    ylim(app.UIAxesL,[(cents(maybe(i),2)-100) (cents(maybe(i),2)+100)])
                    
                    
                    imshow(app.orig_img,'Parent',app.UIAxesR);
                    hold(app.UIAxesR,'on');
                    plot(cents(maybe(i),1),cents(maybe(i),2),'y*','Parent',app.UIAxesR)
                    xlim(app.UIAxesR,[(cents(maybe(i),1)-100) (cents(maybe(i),1)+100)])
                    ylim(app.UIAxesR,[(cents(maybe(i),2)-100) (cents(maybe(i),2)+100)])
                
                    uiwait(app.UIFigure);
                    
                    if app.notfiber
                        removeidx = Rprop(maybe(i),7);
                        tempmask(removeidx.PixelIdxList{1,1}) = 0;
                    end
                    hold(app.UIAxesL,'off');
                    hold(app.UIAxesR,'off');
                end
                


            app.SortingAxesPanel.Visible = 'off';
            
            hiprobnon = ~ismember(nonfiberindex,maybe);  % Find the index of regions that are nonfiber with high probability
            hiprobnonidx = nonzeros(hiprobnon.*nonfiberindex);
            for i = 1:length(hiprobnonidx)
               hiprobnonprop = Rprop(hiprobnonidx(i),7);
               tempmask(hiprobnonprop.PixelIdxList{1,1}) = 0;
            end
            
            imshow(flattenMaskOverlay(app.orig_img,logical(tempmask),0.5,'w'),'Parent',app.UIAxes);
            
            rgb_label = label2rgb(tempmask,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff');
            
            app.FiberPredictionControlPanel.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            
        end

        % Button pushed function: AcceptSegmentationButton
        function AcceptSegmentationButtonPushed(app, event)
            label = bwlabel(~logical(app.bw_obj),4);
            rgb_label = label2rgb(label,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff');
            app.SegmentationParameters.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualSegmentationButton.Enable ='on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
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
            app.ManualSortingButton.Enable = 'off';
            uiresume(app.UIFigure);
            
            
        end

        % Button pushed function: RemoveObjectsButton
        function RemoveObjectsButtonPushed(app, event)
            app.RemoveObjectsButton.Enable = 'off';
            app.MergeObjectsButton.Enable = 'off';
            app.FinishManualFilteringButton.Enable = 'on';
            label = bwlabel(app.bw_obj,4);
            bw_pos = app.bw_obj;
            bw_all = app.bw_obj;
            app.num_obj = max(max(label));
            app.done = 0;
            pon = true(app.num_obj,1); % logical ones array to determine if a region has been selected
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            while ~app.done
                app.Prompt.Text = 'Click on regions for removal';
                phand = drawpoint(app.UIAxes,'Color','w');
                %wait(phand)
                pos = round(phand.Position);
                xp = pos(1);
                yp = pos(2);
                delete(phand)
                if ~app.done
                    % Continue if clicked point is out of bounds
                    xp_is_valid = xp > 0 & xp <= size(label,2);
                    yp_is_valid = yp > 0 & yp <= size(label,1);
                    if ~xp_is_valid || ~yp_is_valid
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
                    
                    flat_img = flattenMaskOverlay(app.orig_img,bw_all,0.1,'w');
                    flat_img = flattenMaskOverlay(flat_img,bw_pos,0.5,'w');
                    imshow(flat_img,'Parent',app.UIAxes);
                end
            end
            app.bw_obj = bw_pos;
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            label = bwlabel(app.bw_obj,4);
            rgb_label = label2rgb(label,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff');
            app.ManualFilterControls.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            app.Prompt.Text = '';
            app.RemoveObjectsButton.Enable = 'on';
            app.MergeObjectsButton.Enable = 'on';
        end

        % Button pushed function: FinishManualFilteringButton
        function FinishManualFilteringButtonPushed(app, event)
            app.Prompt.Text = 'Click anywhere to continue';
            app.FinishManualFilteringButton.Enable = 'off';
            app.done = 1;
            
        end

        % Button pushed function: CalculateFiberProperties
        function CalculateFiberPropertiesPushed(app, event)
            app.WritetoExcelButton.Enable = 'on';
            app.pix_size = app.PixelSizeumpixelEditField.Value;
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

        % Button pushed function: DoneButton
        function DoneButtonPushed(app, event)
            app.Prompt.Text = '';
            app.PropertiesControlPanel.Visible = 'off';
            app.PropertiesPanel.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
        end

        % Button pushed function: WritetoExcelButton
        function WritetoExcelButtonPushed(app, event)
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
            app.Prompt.Text = 'Write to Excel done';
            app.props = 0;
            cd(app.Files{3})
            
        end

        % Value changed function: SortingThresholdSlider
        function SortingThresholdSliderValueChanged(app, event)
            value = app.SortingThresholdSlider.Value;
            app.SortingThresholdSlider.Value = round(value,1);
        end

        % Button pushed function: CalculateCentralNuclei
        function CalculateCentralNucleiPushed(app, event)
            app.Prompt.Text = '';
            app.CalculateCentralNuclei.Enable = 'off';
            app.DoneButton_CNF.Enable = 'off';
            app.pix_size = app.PixelSizeumpixelEditField_2.Value;
            min_nuc_pix = app.MinimumNucleusSizeum2EditField.Value/(app.pix_size^2);
            border = app.DistancefromborderEditField.Value;
            border_pix = border/app.pix_size;
            label_org = bwlabel(app.bw_obj,4);
            num_fib = max(max(label_org));
            
            % Create border region
            inv_img = imcomplement(app.bw_obj);
            dist = bwdist(inv_img);
            inv_brd = (dist > border_pix);
            brd_img = imcomplement(inv_brd);
            
            % Define Nuclei
            nuc_obj = app.orig_img_multispectral(:,:,str2double(app.NucleiColorDropDown.Value));
            se = strel('disk',12);
            tophatFiltered = imtophat(nuc_obj,se);
            nuc_fil = imadjust(tophatFiltered);
            threshes = multithresh(nuc_fil,10);
            app.thresh_nuc = double(threshes(2))/255;
            
            app.ThresholdCNF.Enable = 'on';
            app.AdjustCNF.Enable = 'on';
            app.AcceptCNF.Enable = 'on';
            app.ThresholdCNF.Value = app.thresh_nuc*255;
            
            app.CNF_Adj = 1;
            
            while app.CNF_Adj
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
                cnf_img = app.bw_obj.*0.3;
                cnf_img(vertcat(app.fprop(app.cen_nuc).PixelIdxList)) = 1;
            
                % Create image showing nuclei and border region
                %comb_img = zeros([size(brd_img) 3]);
                %comb_img(:,:,1) = uint8(brd_img).*255;
                %comb_img(:,:,3) = uint8(nuc_bw).*255;
            
                % Show plots
                %imshow(app.orig_img,'Parent',app.UIAxes2_TL)
                %imshow(nuc_fil,'Parent',app.UIAxes2_TR)
                %imshow(comb_img,'Parent',app.UIAxes2_BL)
                %imshow(cnf_img,'Parent',app.UIAxes2_BR)
                pos_img = flattenMaskOverlay(cnf_img,nuc_bw,0.6,'b');
                imshow(pos_img,'Parent',app.CNFAxes);
                uiwait(app.UIFigure);
            end
            
        end

        % Button pushed function: CNFExcelWrite
        function CNFExcelWriteButtonPushed(app, event)
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
            
            data = zeros(app.num_obj,3);
            farea = [app.fprop.Area]';
            farea = farea.*(app.pix_size^2);
            data = cat(2,farea,app.cen_pix.*app.pix_size^2,app.cen_nuc) ;
            out = cat(1,header,num2cell(data));
            writecell(out,[app.Files{4} '_Properties.xlsx'],'Range','I1');
            app.Prompt.Text = 'Write to Excel done';
            cd(app.Files{3})
            app.fprop = 0;
            app.cen_nuc = 0;
            app.cen_pix = 0;
        end

        % Button pushed function: DoneButton_CNF
        function DoneButton_CNFPushed(app, event)
            app.Prompt.Text = '';
            app.CNFControlPanel.Visible = 'off';
            app.CNFPanel.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
        end

        % Button pushed function: CalculateFiberTyping
        function CalculateFiberTypingButtonPushed(app, event)
            app.CalculateFiberTyping.Enable = 'off';
            app.PixelSizeFiberType.Enable = 'off';
            app.FiberTypeColorDropDown.Enable = 'off';
            app.DoneFT.Enable = 'off';
            app.WritetoExcelFT.Enable = 'off';
            app.pix_size = app.PixelSizeFiberType.Value;
            pix_area = app.pix_size^2;
            
            % Create Labelled Fibers
            label = bwlabel(app.bw_obj,4);
            app.num_obj = max(max(label));
            
            % Threshold Fiber Types
            fti = app.orig_img_multispectral(:,:,str2double(app.FiberTypeColorDropDown.Value));
            threshes = multithresh(fti,10);
            app.cutoff_avg = threshes(2);
            
            % Fiber Properties
            rprop = regionprops(label,fti,'MeanIntensity','Centroid','Area','PixelIdxList');
            app.ave_g = [rprop.MeanIntensity];
            app.areas = [rprop.Area];
            cents = cat(1,rprop.Centroid);
            
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
                
                imshow(app.orig_img,'Parent',app.FTAxesL);
                imshow(img_out,'Parent',app.FTAxesR);
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
            app.DoneFT.Enable = 'on';
            uiresume(app.UIFigure);
        end

        % Button pushed function: WritetoExcelFT
        function WritetoExcelFTButtonPushed(app, event)
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
            app.Prompt.Text = 'Write to Excel done';
            cd(app.Files{3})
            app.ponf = 0;
            app.ave_g = 0;
            app.areas = 0;
        end

        % Button pushed function: DoneFT
        function DoneFTButtonPushed(app, event)
            app.Prompt.Text = '';
            app.FiberTypingControlPanel.Visible = 'off';
            app.FiberTypingPanel.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
        end

        % Button pushed function: AdjustCNF
        function AdjustCNFButtonPushed(app, event)
            app.thresh_nuc = app.ThresholdCNF.Value/255;
            uiresume(app.UIFigure);
        end

        % Button pushed function: AcceptCNF
        function AcceptCNFButtonPushed(app, event)
            app.CNF_Adj = 0;
            app.ThresholdCNF.Enable = 'off';
            app.AdjustCNF.Enable = 'off';
            app.AcceptCNF.Enable = 'off';
            app.CalculateCentralNuclei.Enable = 'on';
            app.CNFExcelWrite.Enable = 'on';
            app.DoneButton_CNF.Enable = 'on';
            uiresume(app.UIFigure);
        end

        % Button pushed function: CalculateNonfiberObjects
        function CalculateNonfiberObjectsButtonPushed(app, event)
            img_org = app.orig_img_multispectral;
            app.CalculateNonfiberObjects.Enable = 'off';
            app.DoneNonfiber.Enable = 'off';
            app.pix_size = app.PixelSizeNonfiber.Value;
            ch_obj = img_org(:,:,str2double(app.NonfiberObjectColor.Value));
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
            
            while app.Obj_Adj
                ch_bw = imbinarize(ch_fil,app.thresh_nf);
                imshow(ch_bw,'Parent',app.NonfiberAxes);
                
                uiwait(app.UIFigure);
                
                if app.Obj_Adj
                    app.thresh_nf = app.NonfiberThreshold.Value/255;
                end
                
            end
            
            label = bwlabel(ch_bw,4);
            nfprops = regionprops(label,'Centroid','Area');
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
            app.Prompt.Text = 'Write to Excel done';
            cd(app.Files{3})
            
        end

        % Button pushed function: DoneNonfiber
        function DoneNonfiberButtonPushed(app, event)
            app.thresh_nf = 0;
            app.num_nf = 0;
            app.nf_data = 0;
            app.Prompt.Text = '';
            app.NonfiberPanel.Visible = 'off';
            app.NonfiberControlPanel.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            app.SelectFileButton.Enable = 'on';
            
        end

        % Button pushed function: InitialSegmentationButton
        function InitialSegmentationButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.SegmentationParameters.Visible = 'on';
        end

        % Button pushed function: ManualSegmentationButton
        function ManualSegmentationButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.DrawLineControls.Visible = 'on';
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imbinarize(graymask,0.99);
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,1,'w'),'Parent',app.UIAxes);
               
        end

        % Button pushed function: FiberPredictionButton
        function FiberPredictionButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.FiberPredictionControlPanel.Visible = 'on';
            % acquire mask and show over image
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imcomplement(imbinarize(graymask,0.99));
            app.bw_obj = imclearborder(app.bw_obj,4);
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            app.FilterButton.Enable = 'on';
            app.SortingThresholdSlider.Enable = 'on';
        end

        % Button pushed function: ManualFiberFilterButton
        function ManualFiberFilterButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.ManualFilterControls.Visible = 'on';
            app.RemoveObjectsButton.Enable = 'on';
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imcomplement(imbinarize(graymask,0.99));
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
        end

        % Button pushed function: FiberPropertiesButton
        function FiberPropertiesButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.PropertiesControlPanel.Visible = 'on';
            app.PropertiesPanel.Visible = 'on';
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imcomplement(imbinarize(graymask,0.99));
            app.bw_obj = imclearborder(app.bw_obj,4);
               
        end

        % Button pushed function: CentralNucleiButton
        function CentralNucleiButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CNFPanel.Visible = 'on';
            app.CNFControlPanel.Visible = 'on';
            app.ThresholdCNF.Enable = 'off';
            app.AdjustCNF.Enable = 'off';
            app.AcceptCNF.Enable = 'off';
            app.CNFExcelWrite.Enable = 'off';
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imcomplement(imbinarize(graymask,0.99));
        end

        % Button pushed function: FiberTypingButton
        function FiberTypingButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.FiberTypingPanel.Visible = 'on';
            app.FiberTypingControlPanel.Visible = 'on';
            app.PixelSizeFiberType.Enable = 'on';
            app.FiberTypeColorDropDown.Enable = 'on';
            app.WritetoExcelFT.Enable = 'off';
            mask = imread(app.Files{2});
            graymask = rgb2gray(mask);
            app.bw_obj = imcomplement(imbinarize(graymask,0.99));
        end

        % Button pushed function: NonfiberObjectsButton
        function NonfiberObjectsButtonPushed(app, event)
            app.SelectFileButton.Enable = 'off';
            app.InitialSegmentationButton.Enable = 'off';
            app.FiberPredictionButton.Enable = 'off';
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualSegmentationButton.Enable = 'off';
            app.CentralNucleiButton.Enable = 'off';
            app.FiberTypingButton.Enable = 'off';
            app.NonfiberObjectsButton.Enable = 'off';
            app.FiberPropertiesButton.Enable = 'off';
            app.NonfiberPanel.Visible = 'on';
            app.NonfiberControlPanel.Visible = 'on';
            app.NonfiberThreshold.Enable = 'off';
            app.NonfiberAdjust.Enable = 'off';
            app.NonfiberAccept.Enable = 'off';
            app.WritetoExcelNonfiber.Enable = 'off';
        end

        % Button pushed function: DetectValueButton
        function DetectValueButtonPushed(app, event)
            foc = app.FiberOutlineColorDropDown.Value;
            foc = str2double(foc);
            lam = app.orig_img(:,:,foc);  % fiber outline color
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

        % Button pushed function: MergeObjectsButton
        function MergeObjectsButtonPushed(app, event)
            app.RemoveObjectsButton.Enable = 'off';
            app.MergeObjectsButton.Enable = 'off';
            app.FinishManualFilteringButton.Enable = 'on';

            object_labels = bwlabel(app.bw_obj,4);
            bw_selected = CreateEmptyImageBWImageWithAppOrigImgDimensions(app);
            app.done = 0;

            NONE_REGION_SELECTED = -1;
            first_region_to_merge = NONE_REGION_SELECTED;

            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            while ~app.done
                if first_region_to_merge == NONE_REGION_SELECTED
                    app.Prompt.Text = 'Select first region to merge';
                else
                    app.Prompt.Text = 'Select second region to merge to first region or unselect the first region';
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
                        app.bw_obj = MergeObjects(app, object_labels, first_region_to_merge, second_region_to_merge);
 
                        %%%% Reset Local Variables

                        % Two connected components have been merged, so we need to relabel
                        object_labels = bwlabel(app.bw_obj, 4);

                        % Select the newly created object
                        new_object_label = object_labels(yp,xp);
                        bw_selected = object_labels == new_object_label;
                        first_region_to_merge = new_object_label;
                    end

                    % Draw objects and selection
                    flat_img = flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w');
                    flat_img = flattenMaskOverlay(flat_img,bw_selected,0.8,'w');
                    imshow(flat_img,'Parent',app.UIAxes);
                end
            end
            imshow(flattenMaskOverlay(app.orig_img,app.bw_obj,0.5,'w'),'Parent',app.UIAxes);
            object_labels = bwlabel(app.bw_obj,4);
            app.num_obj = max(max(object_labels));
            rgb_label = label2rgb(object_labels,'jet','w','shuffle');
            imwrite(rgb_label,app.Files{2},'tiff')

            app.ManualFilterControls.Visible = 'off';
            app.InitialSegmentationButton.Enable = 'on';
            app.ManualSegmentationButton.Enable = 'on';
            app.FiberPredictionButton.Enable = 'on';
            app.ManualFiberFilterButton.Enable = 'on';
            app.FiberPropertiesButton.Enable = 'on';
            app.CentralNucleiButton.Enable = 'on';
            app.FiberTypingButton.Enable = 'on';
            app.NonfiberObjectsButton.Enable = 'on';
            app.Prompt.Text = '';
            app.RemoveObjectsButton.Enable = 'on';
            app.MergeObjectsButton.Enable = 'on';
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1199 779];
            app.UIFigure.Name = 'MATLAB App';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            xlabel(app.UIAxes, 'X')
            ylabel(app.UIAxes, 'Y')
            app.UIAxes.XColor = 'none';
            app.UIAxes.YColor = 'none';
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Position = [266 9 909 698];

            % Create SelectFileButton
            app.SelectFileButton = uibutton(app.UIFigure, 'push');
            app.SelectFileButton.ButtonPushedFcn = createCallbackFcn(app, @SelectFileButtonPushed, true);
            app.SelectFileButton.Position = [38 671 109 32];
            app.SelectFileButton.Text = 'Select File';

            % Create FilenameLabel
            app.FilenameLabel = uilabel(app.UIFigure);
            app.FilenameLabel.Position = [156 676 130 22];
            app.FilenameLabel.Text = 'Filename';

            % Create SegmentationParameters
            app.SegmentationParameters = uipanel(app.UIFigure);
            app.SegmentationParameters.Visible = 'off';
            app.SegmentationParameters.Position = [18 406 260 245];

            % Create PixelSizeumpixelEditFieldLabel
            app.PixelSizeumpixelEditFieldLabel = uilabel(app.SegmentationParameters);
            app.PixelSizeumpixelEditFieldLabel.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditFieldLabel.Position = [16 200 114 22];
            app.PixelSizeumpixelEditFieldLabel.Text = 'Pixel Size (um/pixel)';

            % Create PixelSizeField
            app.PixelSizeField = uieditfield(app.SegmentationParameters, 'numeric');
            app.PixelSizeField.Position = [145 200 100 22];

            % Create SegmentButton
            app.SegmentButton = uibutton(app.SegmentationParameters, 'push');
            app.SegmentButton.ButtonPushedFcn = createCallbackFcn(app, @SegmentButtonPushed, true);
            app.SegmentButton.Position = [144 65 100 22];
            app.SegmentButton.Text = 'Segment';

            % Create FiberOutlineColorDropDownLabel
            app.FiberOutlineColorDropDownLabel = uilabel(app.SegmentationParameters);
            app.FiberOutlineColorDropDownLabel.HorizontalAlignment = 'right';
            app.FiberOutlineColorDropDownLabel.Position = [24 158 106 22];
            app.FiberOutlineColorDropDownLabel.Text = 'Fiber Outline Color';

            % Create FiberOutlineColorDropDown
            app.FiberOutlineColorDropDown = uidropdown(app.SegmentationParameters);
            app.FiberOutlineColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.FiberOutlineColorDropDown.ItemsData = {'1', '2', '3'};
            app.FiberOutlineColorDropDown.Position = [145 158 100 22];
            app.FiberOutlineColorDropDown.Value = '1';

            % Create AcceptSegmentationButton
            app.AcceptSegmentationButton = uibutton(app.SegmentationParameters, 'push');
            app.AcceptSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @AcceptSegmentationButtonPushed, true);
            app.AcceptSegmentationButton.Enable = 'off';
            app.AcceptSegmentationButton.Position = [81 33 100 22];
            app.AcceptSegmentationButton.Text = 'Accept';

            % Create SegmentationThresholdSliderLabel
            app.SegmentationThresholdSliderLabel = uilabel(app.SegmentationParameters);
            app.SegmentationThresholdSliderLabel.HorizontalAlignment = 'center';
            app.SegmentationThresholdSliderLabel.Position = [13 101 80 43];
            app.SegmentationThresholdSliderLabel.Text = {'Segmentation'; 'Threshold'};

            % Create SegmentationThresholdSlider
            app.SegmentationThresholdSlider = uislider(app.SegmentationParameters);
            app.SegmentationThresholdSlider.Limits = [0 50];
            app.SegmentationThresholdSlider.ValueChangedFcn = createCallbackFcn(app, @SegmentationThresholdSliderValueChanged, true);
            app.SegmentationThresholdSlider.Position = [101 128 139 3];

            % Create DetectValueButton
            app.DetectValueButton = uibutton(app.SegmentationParameters, 'push');
            app.DetectValueButton.ButtonPushedFcn = createCallbackFcn(app, @DetectValueButtonPushed, true);
            app.DetectValueButton.Position = [24 65 100 22];
            app.DetectValueButton.Text = 'Detect Value';

            % Create Prompt
            app.Prompt = uilabel(app.UIFigure);
            app.Prompt.HorizontalAlignment = 'center';
            app.Prompt.Position = [257 702 824 22];
            app.Prompt.Text = '';

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.HorizontalAlignment = 'left';
            app.Image.Position = [8 706 52 74];
            app.Image.ImageSource = 'screenshot.png';

            % Create SMASHLabel
            app.SMASHLabel = uilabel(app.UIFigure);
            app.SMASHLabel.FontName = 'Century Gothic';
            app.SMASHLabel.FontSize = 20;
            app.SMASHLabel.Position = [72 733 72 26];
            app.SMASHLabel.Text = 'SMASH';

            % Create Image2
            app.Image2 = uiimage(app.UIFigure);
            app.Image2.HorizontalAlignment = 'right';
            app.Image2.VerticalAlignment = 'top';
            app.Image2.Position = [1088 723 122 45];
            app.Image2.ImageSource = 'LabLogo.png';

            % Create DrawLineControls
            app.DrawLineControls = uipanel(app.UIFigure);
            app.DrawLineControls.Visible = 'off';
            app.DrawLineControls.Position = [38 470 153 183];

            % Create DrawLineButton
            app.DrawLineButton = uibutton(app.DrawLineControls, 'push');
            app.DrawLineButton.ButtonPushedFcn = createCallbackFcn(app, @DrawLineButtonPushed, true);
            app.DrawLineButton.Position = [28 137 100 22];
            app.DrawLineButton.Text = 'Draw Line';

            % Create DoneDrawingButton
            app.DoneDrawingButton = uibutton(app.DrawLineControls, 'push');
            app.DoneDrawingButton.ButtonPushedFcn = createCallbackFcn(app, @DoneDrawingButtonPushed, true);
            app.DoneDrawingButton.Enable = 'off';
            app.DoneDrawingButton.Position = [28 82 100 22];
            app.DoneDrawingButton.Text = 'Done Drawing';

            % Create FinishSegmentingButton
            app.FinishSegmentingButton = uibutton(app.DrawLineControls, 'push');
            app.FinishSegmentingButton.ButtonPushedFcn = createCallbackFcn(app, @FinishSegmentingButtonPushed, true);
            app.FinishSegmentingButton.Position = [21 28 115 22];
            app.FinishSegmentingButton.Text = 'Finish Segmenting';

            % Create SortingAxesPanel
            app.SortingAxesPanel = uipanel(app.UIFigure);
            app.SortingAxesPanel.Title = 'Panel4';
            app.SortingAxesPanel.Visible = 'off';
            app.SortingAxesPanel.Position = [285 24 890 679];

            % Create UIAxesL
            app.UIAxesL = uiaxes(app.SortingAxesPanel);
            xlabel(app.UIAxesL, 'X')
            ylabel(app.UIAxesL, 'Y')
            app.UIAxesL.PlotBoxAspectRatio = [1 1.04306220095694 1];
            app.UIAxesL.XColor = 'none';
            app.UIAxesL.YColor = 'none';
            app.UIAxesL.Position = [12 188 412 438];

            % Create UIAxesR
            app.UIAxesR = uiaxes(app.SortingAxesPanel);
            xlabel(app.UIAxesR, 'X')
            ylabel(app.UIAxesR, 'Y')
            app.UIAxesR.PlotBoxAspectRatio = [1 1.12082262210797 1];
            app.UIAxesR.XColor = 'none';
            app.UIAxesR.YColor = 'none';
            app.UIAxesR.Position = [467 188 412 438];

            % Create YesButton
            app.YesButton = uibutton(app.SortingAxesPanel, 'push');
            app.YesButton.ButtonPushedFcn = createCallbackFcn(app, @YesButtonPushed, true);
            app.YesButton.Position = [178 108 100 22];
            app.YesButton.Text = 'Yes';

            % Create NoButton
            app.NoButton = uibutton(app.SortingAxesPanel, 'push');
            app.NoButton.ButtonPushedFcn = createCallbackFcn(app, @NoButtonPushed, true);
            app.NoButton.Position = [378 108 100 22];
            app.NoButton.Text = 'No';

            % Create MarkasfiberLabel
            app.MarkasfiberLabel = uilabel(app.SortingAxesPanel);
            app.MarkasfiberLabel.HorizontalAlignment = 'center';
            app.MarkasfiberLabel.Position = [188 143 290 22];
            app.MarkasfiberLabel.Text = 'Mark as fiber?';

            % Create PropertiesPanel
            app.PropertiesPanel = uipanel(app.UIFigure);
            app.PropertiesPanel.Visible = 'off';
            app.PropertiesPanel.Position = [285 24 890 679];

            % Create FeretAxes
            app.FeretAxes = uiaxes(app.PropertiesPanel);
            title(app.FeretAxes, 'Minimum Feret Diameter (um)')
            app.FeretAxes.PlotBoxAspectRatio = [3.2695652173913 1 1];
            app.FeretAxes.Position = [53 365 799 285];

            % Create FiberSizeAxes
            app.FiberSizeAxes = uiaxes(app.PropertiesPanel);
            title(app.FiberSizeAxes, 'Fiber Area (um^2)')
            app.FiberSizeAxes.PlotBoxAspectRatio = [3.29824561403509 1 1];
            app.FiberSizeAxes.Position = [53 67 799 285];

            % Create PropertiesControlPanel
            app.PropertiesControlPanel = uipanel(app.UIFigure);
            app.PropertiesControlPanel.Visible = 'off';
            app.PropertiesControlPanel.Position = [21 364 251 275];

            % Create WritetoExcelButton
            app.WritetoExcelButton = uibutton(app.PropertiesControlPanel, 'push');
            app.WritetoExcelButton.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelButtonPushed, true);
            app.WritetoExcelButton.Enable = 'off';
            app.WritetoExcelButton.Position = [27 65 100 22];
            app.WritetoExcelButton.Text = 'Write to Excel';

            % Create DoneButton
            app.DoneButton = uibutton(app.PropertiesControlPanel, 'push');
            app.DoneButton.ButtonPushedFcn = createCallbackFcn(app, @DoneButtonPushed, true);
            app.DoneButton.Position = [140 65 100 22];
            app.DoneButton.Text = 'Done';

            % Create DataOutputFolderEditFieldLabel
            app.DataOutputFolderEditFieldLabel = uilabel(app.PropertiesControlPanel);
            app.DataOutputFolderEditFieldLabel.HorizontalAlignment = 'right';
            app.DataOutputFolderEditFieldLabel.Position = [16 169 108 22];
            app.DataOutputFolderEditFieldLabel.Text = 'Data Output Folder';

            % Create FiberPropertiesDataOutputFolder
            app.FiberPropertiesDataOutputFolder = uieditfield(app.PropertiesControlPanel, 'text');
            app.FiberPropertiesDataOutputFolder.Position = [139 169 100 22];

            % Create PixelSizeumpixelEditFieldLabel_2
            app.PixelSizeumpixelEditFieldLabel_2 = uilabel(app.PropertiesControlPanel);
            app.PixelSizeumpixelEditFieldLabel_2.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditFieldLabel_2.Position = [10 218 114 22];
            app.PixelSizeumpixelEditFieldLabel_2.Text = 'Pixel Size (um/pixel)';

            % Create PixelSizeumpixelEditField
            app.PixelSizeumpixelEditField = uieditfield(app.PropertiesControlPanel, 'numeric');
            app.PixelSizeumpixelEditField.Position = [139 218 100 22];

            % Create CalculateFiberProperties
            app.CalculateFiberProperties = uibutton(app.PropertiesControlPanel, 'push');
            app.CalculateFiberProperties.ButtonPushedFcn = createCallbackFcn(app, @CalculateFiberPropertiesPushed, true);
            app.CalculateFiberProperties.Position = [81 119 100 22];
            app.CalculateFiberProperties.Text = 'Calculate';

            % Create FiberPredictionControlPanel
            app.FiberPredictionControlPanel = uipanel(app.UIFigure);
            app.FiberPredictionControlPanel.Visible = 'off';
            app.FiberPredictionControlPanel.Position = [16 395 263 243];

            % Create FilterButton
            app.FilterButton = uibutton(app.FiberPredictionControlPanel, 'push');
            app.FilterButton.ButtonPushedFcn = createCallbackFcn(app, @FilterButtonPushed, true);
            app.FilterButton.Position = [74 103 100 22];
            app.FilterButton.Text = 'Filter';

            % Create ManualSortingButton
            app.ManualSortingButton = uibutton(app.FiberPredictionControlPanel, 'push');
            app.ManualSortingButton.ButtonPushedFcn = createCallbackFcn(app, @ManualSortingButtonPushed, true);
            app.ManualSortingButton.Enable = 'off';
            app.ManualSortingButton.Position = [75 54 100 22];
            app.ManualSortingButton.Text = 'Manual Sorting';

            % Create SortingThresholdHigherrequiresmoremanualsortingLabel
            app.SortingThresholdHigherrequiresmoremanualsortingLabel = uilabel(app.FiberPredictionControlPanel);
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.HorizontalAlignment = 'center';
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.Position = [-28.5 165 209 43];
            app.SortingThresholdHigherrequiresmoremanualsortingLabel.Text = {'Sorting Threshold'; '(Higher requires'; ' more manual sorting)'; ''};

            % Create SortingThresholdSlider
            app.SortingThresholdSlider = uislider(app.FiberPredictionControlPanel);
            app.SortingThresholdSlider.Limits = [0 0.9];
            app.SortingThresholdSlider.MajorTicks = [0 0.5 0.9];
            app.SortingThresholdSlider.ValueChangedFcn = createCallbackFcn(app, @SortingThresholdSliderValueChanged, true);
            app.SortingThresholdSlider.MinorTicks = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
            app.SortingThresholdSlider.Position = [150 195 92 3];

            % Create CNFControlPanel
            app.CNFControlPanel = uipanel(app.UIFigure);
            app.CNFControlPanel.Visible = 'off';
            app.CNFControlPanel.Position = [16 360 259 293];

            % Create PixelSizeumpixelLabel
            app.PixelSizeumpixelLabel = uilabel(app.CNFControlPanel);
            app.PixelSizeumpixelLabel.HorizontalAlignment = 'right';
            app.PixelSizeumpixelLabel.Position = [48 255 58 28];
            app.PixelSizeumpixelLabel.Text = {'Pixel Size'; 'um/pixel'};

            % Create PixelSizeumpixelEditField_2
            app.PixelSizeumpixelEditField_2 = uieditfield(app.CNFControlPanel, 'numeric');
            app.PixelSizeumpixelEditField_2.Position = [121 261 100 22];

            % Create NucleiColorDropDownLabel
            app.NucleiColorDropDownLabel = uilabel(app.CNFControlPanel);
            app.NucleiColorDropDownLabel.HorizontalAlignment = 'right';
            app.NucleiColorDropDownLabel.Position = [39 226 71 22];
            app.NucleiColorDropDownLabel.Text = 'Nuclei Color';

            % Create NucleiColorDropDown
            app.NucleiColorDropDown = uidropdown(app.CNFControlPanel);
            app.NucleiColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.NucleiColorDropDown.ItemsData = {'1', '2', '3'};
            app.NucleiColorDropDown.Position = [125 226 100 22];
            app.NucleiColorDropDown.Value = '1';

            % Create CalculateCentralNuclei
            app.CalculateCentralNuclei = uibutton(app.CNFControlPanel, 'push');
            app.CalculateCentralNuclei.ButtonPushedFcn = createCallbackFcn(app, @CalculateCentralNucleiPushed, true);
            app.CalculateCentralNuclei.Position = [85 66 100 22];
            app.CalculateCentralNuclei.Text = 'Calculate';

            % Create CNFExcelWrite
            app.CNFExcelWrite = uibutton(app.CNFControlPanel, 'push');
            app.CNFExcelWrite.ButtonPushedFcn = createCallbackFcn(app, @CNFExcelWriteButtonPushed, true);
            app.CNFExcelWrite.Position = [24 33 100 22];
            app.CNFExcelWrite.Text = 'Write To Excel';

            % Create DoneButton_CNF
            app.DoneButton_CNF = uibutton(app.CNFControlPanel, 'push');
            app.DoneButton_CNF.ButtonPushedFcn = createCallbackFcn(app, @DoneButton_CNFPushed, true);
            app.DoneButton_CNF.Position = [142 33 100 22];
            app.DoneButton_CNF.Text = 'Done';

            % Create DistancefromborderEditFieldLabel
            app.DistancefromborderEditFieldLabel = uilabel(app.CNFControlPanel);
            app.DistancefromborderEditFieldLabel.HorizontalAlignment = 'right';
            app.DistancefromborderEditFieldLabel.Position = [15 194 118 22];
            app.DistancefromborderEditFieldLabel.Text = 'Distance from border';

            % Create DistancefromborderEditField
            app.DistancefromborderEditField = uieditfield(app.CNFControlPanel, 'numeric');
            app.DistancefromborderEditField.Position = [148 194 100 22];

            % Create MinimumNucleusSizeum2EditFieldLabel
            app.MinimumNucleusSizeum2EditFieldLabel = uilabel(app.CNFControlPanel);
            app.MinimumNucleusSizeum2EditFieldLabel.HorizontalAlignment = 'right';
            app.MinimumNucleusSizeum2EditFieldLabel.Position = [7 157 128 28];
            app.MinimumNucleusSizeum2EditFieldLabel.Text = {'Minimum Nucleus Size'; '(um^2)'};

            % Create MinimumNucleusSizeum2EditField
            app.MinimumNucleusSizeum2EditField = uieditfield(app.CNFControlPanel, 'numeric');
            app.MinimumNucleusSizeum2EditField.Position = [150 163 100 22];

            % Create DataOutputFolderEditField_2Label
            app.DataOutputFolderEditField_2Label = uilabel(app.CNFControlPanel);
            app.DataOutputFolderEditField_2Label.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_2Label.Position = [16 128 108 22];
            app.DataOutputFolderEditField_2Label.Text = 'Data Output Folder';

            % Create CentralNucleiDataOutputFolder
            app.CentralNucleiDataOutputFolder = uieditfield(app.CNFControlPanel, 'text');
            app.CentralNucleiDataOutputFolder.Position = [139 128 109 22];

            % Create FiberTypingControlPanel
            app.FiberTypingControlPanel = uipanel(app.UIFigure);
            app.FiberTypingControlPanel.Visible = 'off';
            app.FiberTypingControlPanel.Position = [16 367 256 293];

            % Create PixelSizeumpixelEditField_3Label
            app.PixelSizeumpixelEditField_3Label = uilabel(app.FiberTypingControlPanel);
            app.PixelSizeumpixelEditField_3Label.HorizontalAlignment = 'right';
            app.PixelSizeumpixelEditField_3Label.Position = [45 236 58 28];
            app.PixelSizeumpixelEditField_3Label.Text = {'Pixel Size'; '(um/pixel)'};

            % Create PixelSizeFiberType
            app.PixelSizeFiberType = uieditfield(app.FiberTypingControlPanel, 'numeric');
            app.PixelSizeFiberType.Position = [118 242 100 22];

            % Create DataOutputFolderEditField_3Label
            app.DataOutputFolderEditField_3Label = uilabel(app.FiberTypingControlPanel);
            app.DataOutputFolderEditField_3Label.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_3Label.Position = [19 144 108 22];
            app.DataOutputFolderEditField_3Label.Text = 'Data Output Folder';

            % Create FiberTypingDataOutputFolder
            app.FiberTypingDataOutputFolder = uieditfield(app.FiberTypingControlPanel, 'text');
            app.FiberTypingDataOutputFolder.Position = [142 144 100 22];

            % Create FiberTypeColorDropDownLabel
            app.FiberTypeColorDropDownLabel = uilabel(app.FiberTypingControlPanel);
            app.FiberTypeColorDropDownLabel.HorizontalAlignment = 'right';
            app.FiberTypeColorDropDownLabel.Position = [17 197 94 22];
            app.FiberTypeColorDropDownLabel.Text = 'Fiber Type Color';

            % Create FiberTypeColorDropDown
            app.FiberTypeColorDropDown = uidropdown(app.FiberTypingControlPanel);
            app.FiberTypeColorDropDown.Items = {'Red', 'Green', 'Blue'};
            app.FiberTypeColorDropDown.ItemsData = {'1', '2', '3'};
            app.FiberTypeColorDropDown.Position = [126 197 100 22];
            app.FiberTypeColorDropDown.Value = '1';

            % Create CalculateFiberTyping
            app.CalculateFiberTyping = uibutton(app.FiberTypingControlPanel, 'push');
            app.CalculateFiberTyping.ButtonPushedFcn = createCallbackFcn(app, @CalculateFiberTypingButtonPushed, true);
            app.CalculateFiberTyping.Position = [80 90 100 22];
            app.CalculateFiberTyping.Text = 'Calculate';

            % Create WritetoExcelFT
            app.WritetoExcelFT = uibutton(app.FiberTypingControlPanel, 'push');
            app.WritetoExcelFT.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelFTButtonPushed, true);
            app.WritetoExcelFT.Enable = 'off';
            app.WritetoExcelFT.Position = [26 45 100 22];
            app.WritetoExcelFT.Text = 'Write to Excel';

            % Create DoneFT
            app.DoneFT = uibutton(app.FiberTypingControlPanel, 'push');
            app.DoneFT.ButtonPushedFcn = createCallbackFcn(app, @DoneFTButtonPushed, true);
            app.DoneFT.Position = [145 45 100 22];
            app.DoneFT.Text = 'Done';

            % Create FiberTypingPanel
            app.FiberTypingPanel = uipanel(app.UIFigure);
            app.FiberTypingPanel.Visible = 'off';
            app.FiberTypingPanel.Position = [285 24 882 679];

            % Create FTAxesL
            app.FTAxesL = uiaxes(app.FiberTypingPanel);
            app.FTAxesL.PlotBoxAspectRatio = [1.06832298136646 1 1];
            app.FTAxesL.XColor = 'none';
            app.FTAxesL.YColor = 'none';
            app.FTAxesL.Position = [22 318 414 332];

            % Create FTAxesR
            app.FTAxesR = uiaxes(app.FiberTypingPanel);
            app.FTAxesR.PlotBoxAspectRatio = [1.06832298136646 1 1];
            app.FTAxesR.XColor = 'none';
            app.FTAxesR.YColor = 'none';
            app.FTAxesR.Position = [450 318 416 335];

            % Create FThistL
            app.FThistL = uiaxes(app.FiberTypingPanel);
            app.FThistL.PlotBoxAspectRatio = [1.83030303030303 1 1];
            app.FThistL.Position = [22 112 402 207];

            % Create FThistR
            app.FThistR = uiaxes(app.FiberTypingPanel);
            app.FThistR.PlotBoxAspectRatio = [1.91358024691358 1 1];
            app.FThistR.Position = [450 115 402 204];

            % Create ThresholdEditFieldLabel
            app.ThresholdEditFieldLabel = uilabel(app.FiberTypingPanel);
            app.ThresholdEditFieldLabel.HorizontalAlignment = 'right';
            app.ThresholdEditFieldLabel.Enable = 'off';
            app.ThresholdEditFieldLabel.Position = [123 70 59 22];
            app.ThresholdEditFieldLabel.Text = 'Threshold';

            % Create ThresholdEditField
            app.ThresholdEditField = uieditfield(app.FiberTypingPanel, 'numeric');
            app.ThresholdEditField.Enable = 'off';
            app.ThresholdEditField.Position = [197 70 100 22];

            % Create AdjustButton
            app.AdjustButton = uibutton(app.FiberTypingPanel, 'push');
            app.AdjustButton.ButtonPushedFcn = createCallbackFcn(app, @AdjustButtonPushed, true);
            app.AdjustButton.Enable = 'off';
            app.AdjustButton.Position = [336 70 100 22];
            app.AdjustButton.Text = 'Adjust';

            % Create AcceptButton
            app.AcceptButton = uibutton(app.FiberTypingPanel, 'push');
            app.AcceptButton.ButtonPushedFcn = createCallbackFcn(app, @AcceptButtonPushed, true);
            app.AcceptButton.Enable = 'off';
            app.AcceptButton.Position = [467 70 100 22];
            app.AcceptButton.Text = 'Accept';

            % Create CNFPanel
            app.CNFPanel = uipanel(app.UIFigure);
            app.CNFPanel.Visible = 'off';
            app.CNFPanel.Position = [285 24 882 679];

            % Create CNFAxes
            app.CNFAxes = uiaxes(app.CNFPanel);
            app.CNFAxes.PlotBoxAspectRatio = [1.34971644612476 1 1];
            app.CNFAxes.XColor = 'none';
            app.CNFAxes.YColor = 'none';
            app.CNFAxes.Position = [22 115 743 555];

            % Create ThresholdEditField_2Label
            app.ThresholdEditField_2Label = uilabel(app.CNFPanel);
            app.ThresholdEditField_2Label.HorizontalAlignment = 'right';
            app.ThresholdEditField_2Label.Position = [120 66 59 22];
            app.ThresholdEditField_2Label.Text = 'Threshold';

            % Create ThresholdCNF
            app.ThresholdCNF = uieditfield(app.CNFPanel, 'numeric');
            app.ThresholdCNF.Position = [194 66 100 22];

            % Create AdjustCNF
            app.AdjustCNF = uibutton(app.CNFPanel, 'push');
            app.AdjustCNF.ButtonPushedFcn = createCallbackFcn(app, @AdjustCNFButtonPushed, true);
            app.AdjustCNF.Position = [338 67 100 22];
            app.AdjustCNF.Text = 'Adjust';

            % Create AcceptCNF
            app.AcceptCNF = uibutton(app.CNFPanel, 'push');
            app.AcceptCNF.ButtonPushedFcn = createCallbackFcn(app, @AcceptCNFButtonPushed, true);
            app.AcceptCNF.Position = [477 67 100 22];
            app.AcceptCNF.Text = 'Accept';

            % Create NonfiberPanel
            app.NonfiberPanel = uipanel(app.UIFigure);
            app.NonfiberPanel.Visible = 'off';
            app.NonfiberPanel.Position = [281 24 886 679];

            % Create NonfiberAxes
            app.NonfiberAxes = uiaxes(app.NonfiberPanel);
            xlabel(app.NonfiberAxes, 'X')
            ylabel(app.NonfiberAxes, 'Y')
            app.NonfiberAxes.PlotBoxAspectRatio = [1.35976789168279 1 1];
            app.NonfiberAxes.XColor = 'none';
            app.NonfiberAxes.YColor = 'none';
            app.NonfiberAxes.Position = [22 117 844 553];

            % Create ThresholdEditField_2Label_2
            app.ThresholdEditField_2Label_2 = uilabel(app.NonfiberPanel);
            app.ThresholdEditField_2Label_2.HorizontalAlignment = 'right';
            app.ThresholdEditField_2Label_2.Position = [116 84 59 22];
            app.ThresholdEditField_2Label_2.Text = 'Threshold';

            % Create NonfiberThreshold
            app.NonfiberThreshold = uieditfield(app.NonfiberPanel, 'numeric');
            app.NonfiberThreshold.Position = [190 84 100 22];

            % Create NonfiberAdjust
            app.NonfiberAdjust = uibutton(app.NonfiberPanel, 'push');
            app.NonfiberAdjust.ButtonPushedFcn = createCallbackFcn(app, @NonfiberAdjustButtonPushed, true);
            app.NonfiberAdjust.Position = [333 85 100 22];
            app.NonfiberAdjust.Text = 'Adjust';

            % Create NonfiberAccept
            app.NonfiberAccept = uibutton(app.NonfiberPanel, 'push');
            app.NonfiberAccept.ButtonPushedFcn = createCallbackFcn(app, @NonfiberAcceptButtonPushed, true);
            app.NonfiberAccept.Position = [467 84 100 22];
            app.NonfiberAccept.Text = 'Accept';

            % Create NonfiberControlPanel
            app.NonfiberControlPanel = uipanel(app.UIFigure);
            app.NonfiberControlPanel.Visible = 'off';
            app.NonfiberControlPanel.Position = [20 359 252 301];

            % Create ObjectColorDropDownLabel
            app.ObjectColorDropDownLabel = uilabel(app.NonfiberControlPanel);
            app.ObjectColorDropDownLabel.HorizontalAlignment = 'right';
            app.ObjectColorDropDownLabel.Position = [38 197 72 22];
            app.ObjectColorDropDownLabel.Text = 'Object Color';

            % Create NonfiberObjectColor
            app.NonfiberObjectColor = uidropdown(app.NonfiberControlPanel);
            app.NonfiberObjectColor.Items = {'Red', 'Green', 'Blue'};
            app.NonfiberObjectColor.ItemsData = {'1', '2', '3'};
            app.NonfiberObjectColor.Position = [125 197 100 22];
            app.NonfiberObjectColor.Value = '1';

            % Create DataOutputFolderEditField_3Label_2
            app.DataOutputFolderEditField_3Label_2 = uilabel(app.NonfiberControlPanel);
            app.DataOutputFolderEditField_3Label_2.HorizontalAlignment = 'right';
            app.DataOutputFolderEditField_3Label_2.Position = [5 139 108 22];
            app.DataOutputFolderEditField_3Label_2.Text = 'Data Output Folder';

            % Create NonfiberObjectsDataOutputFolder
            app.NonfiberObjectsDataOutputFolder = uieditfield(app.NonfiberControlPanel, 'text');
            app.NonfiberObjectsDataOutputFolder.Position = [128 139 100 22];

            % Create CalculateNonfiberObjects
            app.CalculateNonfiberObjects = uibutton(app.NonfiberControlPanel, 'push');
            app.CalculateNonfiberObjects.ButtonPushedFcn = createCallbackFcn(app, @CalculateNonfiberObjectsButtonPushed, true);
            app.CalculateNonfiberObjects.Position = [82 69 100 22];
            app.CalculateNonfiberObjects.Text = 'Calculate';

            % Create WritetoExcelNonfiber
            app.WritetoExcelNonfiber = uibutton(app.NonfiberControlPanel, 'push');
            app.WritetoExcelNonfiber.ButtonPushedFcn = createCallbackFcn(app, @WritetoExcelNonfiberButtonPushed, true);
            app.WritetoExcelNonfiber.Position = [20 26 100 22];
            app.WritetoExcelNonfiber.Text = 'Write to Excel';

            % Create DoneNonfiber
            app.DoneNonfiber = uibutton(app.NonfiberControlPanel, 'push');
            app.DoneNonfiber.ButtonPushedFcn = createCallbackFcn(app, @DoneNonfiberButtonPushed, true);
            app.DoneNonfiber.Position = [144 27 100 22];
            app.DoneNonfiber.Text = 'Done';

            % Create PixelSizeumpixelLabel_2
            app.PixelSizeumpixelLabel_2 = uilabel(app.NonfiberControlPanel);
            app.PixelSizeumpixelLabel_2.HorizontalAlignment = 'right';
            app.PixelSizeumpixelLabel_2.Position = [52 247 58 28];
            app.PixelSizeumpixelLabel_2.Text = {'Pixel Size'; '(um/pixel)'};

            % Create PixelSizeNonfiber
            app.PixelSizeNonfiber = uieditfield(app.NonfiberControlPanel, 'numeric');
            app.PixelSizeNonfiber.Position = [125 253 100 22];

            % Create Toolbar
            app.Toolbar = uipanel(app.UIFigure);
            app.Toolbar.Position = [144 727 937 41];

            % Create InitialSegmentationButton
            app.InitialSegmentationButton = uibutton(app.Toolbar, 'push');
            app.InitialSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @InitialSegmentationButtonPushed, true);
            app.InitialSegmentationButton.Enable = 'off';
            app.InitialSegmentationButton.Position = [8 9 121 22];
            app.InitialSegmentationButton.Text = 'Initial Segmentation';

            % Create ManualSegmentationButton
            app.ManualSegmentationButton = uibutton(app.Toolbar, 'push');
            app.ManualSegmentationButton.ButtonPushedFcn = createCallbackFcn(app, @ManualSegmentationButtonPushed, true);
            app.ManualSegmentationButton.Enable = 'off';
            app.ManualSegmentationButton.Position = [136 9 132 22];
            app.ManualSegmentationButton.Text = 'Manual Segmentation';

            % Create FiberPredictionButton
            app.FiberPredictionButton = uibutton(app.Toolbar, 'push');
            app.FiberPredictionButton.ButtonPushedFcn = createCallbackFcn(app, @FiberPredictionButtonPushed, true);
            app.FiberPredictionButton.Enable = 'off';
            app.FiberPredictionButton.Position = [275 9 100 22];
            app.FiberPredictionButton.Text = 'Fiber Prediction';

            % Create ManualFiberFilterButton
            app.ManualFiberFilterButton = uibutton(app.Toolbar, 'push');
            app.ManualFiberFilterButton.ButtonPushedFcn = createCallbackFcn(app, @ManualFiberFilterButtonPushed, true);
            app.ManualFiberFilterButton.Enable = 'off';
            app.ManualFiberFilterButton.Position = [382 9 116 22];
            app.ManualFiberFilterButton.Text = 'Manual Fiber Filter';

            % Create FiberPropertiesButton
            app.FiberPropertiesButton = uibutton(app.Toolbar, 'push');
            app.FiberPropertiesButton.ButtonPushedFcn = createCallbackFcn(app, @FiberPropertiesButtonPushed, true);
            app.FiberPropertiesButton.Enable = 'off';
            app.FiberPropertiesButton.Position = [505 9 101 22];
            app.FiberPropertiesButton.Text = 'Fiber Properties';

            % Create CentralNucleiButton
            app.CentralNucleiButton = uibutton(app.Toolbar, 'push');
            app.CentralNucleiButton.ButtonPushedFcn = createCallbackFcn(app, @CentralNucleiButtonPushed, true);
            app.CentralNucleiButton.Enable = 'off';
            app.CentralNucleiButton.Position = [612 9 100 22];
            app.CentralNucleiButton.Text = 'Central Nuclei';

            % Create FiberTypingButton
            app.FiberTypingButton = uibutton(app.Toolbar, 'push');
            app.FiberTypingButton.ButtonPushedFcn = createCallbackFcn(app, @FiberTypingButtonPushed, true);
            app.FiberTypingButton.Enable = 'off';
            app.FiberTypingButton.Position = [718 9 100 22];
            app.FiberTypingButton.Text = 'Fiber Typing';

            % Create NonfiberObjectsButton
            app.NonfiberObjectsButton = uibutton(app.Toolbar, 'push');
            app.NonfiberObjectsButton.ButtonPushedFcn = createCallbackFcn(app, @NonfiberObjectsButtonPushed, true);
            app.NonfiberObjectsButton.Enable = 'off';
            app.NonfiberObjectsButton.Position = [823.5 9 105 22];
            app.NonfiberObjectsButton.Text = 'Nonfiber Objects';

            % Create ManualFilterControls
            app.ManualFilterControls = uipanel(app.UIFigure);
            app.ManualFilterControls.Visible = 'off';
            app.ManualFilterControls.Position = [55 474 232 164];

            % Create RemoveObjectsButton
            app.RemoveObjectsButton = uibutton(app.ManualFilterControls, 'push');
            app.RemoveObjectsButton.ButtonPushedFcn = createCallbackFcn(app, @RemoveObjectsButtonPushed, true);
            app.RemoveObjectsButton.Position = [75 113 104 22];
            app.RemoveObjectsButton.Text = 'Remove Objects';

            % Create FinishManualFilteringButton
            app.FinishManualFilteringButton = uibutton(app.ManualFilterControls, 'push');
            app.FinishManualFilteringButton.ButtonPushedFcn = createCallbackFcn(app, @FinishManualFilteringButtonPushed, true);
            app.FinishManualFilteringButton.Enable = 'off';
            app.FinishManualFilteringButton.Position = [56 32 136 22];
            app.FinishManualFilteringButton.Text = 'Finish Manual Filtering';

            % Create MergeObjectsButton
            app.MergeObjectsButton = uibutton(app.ManualFilterControls, 'push');
            app.MergeObjectsButton.ButtonPushedFcn = createCallbackFcn(app, @MergeObjectsButtonPushed, true);
            app.MergeObjectsButton.Position = [75 71 100 22];
            app.MergeObjectsButton.Text = 'Merge Objects';

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