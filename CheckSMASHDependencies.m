function CheckSMASHDependencies
% CheckSMASHDependencies  Checks SMASH Dependencies.
%   CheckSMASHDependencies checks that required add-ons and Bioformats
%   MATLAB toolbox are installed.
%
%   Required add-ons are listed in SMASHAddons.m file. For each add-on,
%   CheckSMASHDependencies checks that add-on is both installed and
%   enabled.
%
%   CheckSMASHDependencies checks that the <a href="matlab:web('https://docs.openmicroscopy.org/bio-formats/6.9.1/users/matlab/index.html')">Bio-Formats MATLAB toolbox</a>
%   is installed and added to the MATLAB path.
%
%   See also SMASHAddons.

    disp("Checking Add-Ons Required for SMASH...");
    isAddonsCheckSuccessful = CheckAddOnInstallations();
    isBioformatsCheckSuccessful = CheckIfBioformatsIsInstalled();
    areAllDependenciesInstalled = isAddonsCheckSuccessful && isBioformatsCheckSuccessful;
    PrintSummary(areAllDependenciesInstalled);
end

function isSuccess = CheckAddOnInstallations()
% CheckAddOnInstallations  Checks add-on installations.
%   For each add-on listed in SMASHAddons, CheckAddOnInstallations
%   outputs whether or not that add-on is installed and enabled.
%
%   CheckAddOnInstallations() returns:
%     0 if any add-on listed in SMASHAddons is not installed or is not
%       enabled.
%     1 if all add-ons listed in SMASHAddons are installed and enabled.
%
%   See also SMASHAddons.

    successCount = 0;
    for addon = SMASHAddons()
        successCount = successCount + CheckAddOnInstallation(addon);
    end
    isSuccess = successCount == size(SMASHAddons(), 2);
end

function isSuccess = CheckAddOnInstallation(requiredAddon)
% CheckAddOnInstallation  Checks add-on installation.
%   CheckAddOnInstallation checks if requiredAddon is
%   installed and enabled.
%
%   CheckAddOnInstallation(requiredAddon) returns:
%     0 if requiredAddon is not installed or not enabled.
%     1 if requiredAddon is both installed and enabled.

    fprintf("  Checking if Add-On ""%s"" is installed and enable...", requiredAddon);
    isSuccess = CheckIfAddOnIsInstalledAndEnabled(requiredAddon);
    if isSuccess
        disp('Yes!');
    end
end

function isSuccess = CheckIfAddOnIsInstalledAndEnabled(addonName)
% CheckIfAddOnIsInstalledAndEnabled  Checks if add-on is installed and enabled.
%   CheckIfAddOnIsInstalledAndEnabled checks if add-on is installed and
%   enabled and prints to standard error any detected errors.
%
%   CheckIfAddOnIsInstalledAndEnabled(addonName) returns:
%     0 if addonName is not the name of an installed add-on or if
%       addonName is the name of an installed add-on, but the add-on is
%       disabled.
%     1 if addonName is the name of an installed add-on and that add-on is
%       enabled.
%
%   CheckIfAddOnIsInstalledAndEnabled uses matlab.addons.installedAddons to
%   lookup add-ons.
%
%   See also matlab.addons.installedAddons.

    info = GetAddOnInformation(addonName);
    if ~isempty(info)
       isSuccess = CheckIfAddOnIsEnabled(info);
    else
       fprintf(2, " No%c    Add-On ""%s"" is not installed. Please install it.%c", newline, addonName, newline)
       isSuccess = 0;
    end
end

function info = GetAddOnInformation(addonName)
% GetAddOnInformation  Returns add-ons information.
%
%   GetAddOnInformation(addonName) looks up addonName from the list of
%   installed add-ons and returns a table of strings with these fields:
%
%           Name - Name of the add-on
%        Version - Version of the add-on
%        Enabled - Whether the add-on is enabled
%     Identifier - Unique identifier of the add-on
%
%   The table will be 1 x 4 if a matching add-on is found, empty otherwise.
%
%   See also matlab.addons.installedAddons.

    installedAddons = matlab.addons.installedAddons;
    info = installedAddons(strcmp(installedAddons.Name, addonName), :);
end

function isSuccess = CheckIfAddOnIsEnabled(info)
    if info.Enabled
        isSuccess = 1;
    else
       fprintf(2, " No%c    Add-On ""%s"" is installed but not enabled. Please enable it.%c",newline, info.Name, newline)
        isSuccess = 0;
    end
end

function printSeparator
    disp(repmat('-',1,80))
end

function isInstalled = CheckIfBioformatsIsInstalled()
    fprintf("  Checking if the Bio-Formats MATLAB Toolbox is installed...");
    isInstalled = IsBioformatsInstalled();
    if isInstalled
        disp('Yes!');
    else
        fprintf(2, "No%c", newline);
        fprintf(2, "    The Bio-Formats MATLAB Toolbox could not be found on the search path...%c", newline);
        fprintf(2, "    Please follow the installation instructions here: https://docs.openmicroscopy.org/bio-formats/6.9.1/users/matlab/index.html%c", newline)
    end
end

function isInstalled = IsBioformatsInstalled()
    NAME_IS_M_FILE = 2;
    isInstalled = exist("bfopen", "file") == NAME_IS_M_FILE;
end

function PrintSummary(areAllDependenciesInstalled)
    printSeparator();
    if areAllDependenciesInstalled
        disp("You're all set! All required dependencies have been installed!");
    else
        fprintf(2, "Please install the missing dependencies before running SMASH.%c", newline)
    end
end