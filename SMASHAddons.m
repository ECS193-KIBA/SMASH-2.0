function addons = SMASHAddons()
% SMASHAddons  Returns list of add-ons (by name) required by SMASH.
%   The add-on names correspond to the names contained in the return value
%   of matlab.addons.installedAddons.
%
%   DEVELOPER NOTE:
%
%   Please add any new add-ons that SMASH depends on to the list below. Use
%   matlab.addons.installedAddons to get the name of the new add-on. For
%   cleaner version-control diffs, please use a trailing comma and ellipsis
%   (",...") after the add-on name. For example,
%
%   "NameOfNewAddOn",...
%
%   See also matlab.addons.installedAddons.

    addons = [
        "Curve Fitting Toolbox",...
        "flattenMaskOverlay",...
        "Statistics and Machine Learning Toolbox",...
        "Image Processing Toolbox",...
        "Convert between RGB and Color Names",...
    ];
end

