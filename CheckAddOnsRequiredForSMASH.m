function CheckAddOnsRequiredForSMASH
    disp("Checking Add-Ons Required for SMASH...");
    installedAddons = matlab.addons.installedAddons;
    requiredDependencies = SmashDependencies();
    successCount = CheckIfInstalledAddOnsContainAllRequiredDependencies(installedAddons, requiredDependencies);
    isBioformatsInstalled = CheckIfBioformatsIsInstalled();
    printSeparator();
    if successCount == size(requiredDependencies, 2) && isBioformatsInstalled
        disp("You're all set! All required Add-Ons have been installed!");
    else
        fprintf(2, "Please install the missing Add-On(s) before running SMASH.%c", newline)
    end
    
end

function successCount = CheckIfInstalledAddOnsContainAllRequiredDependencies(installedAddons, requiredDependencies)
    successCount = 0;
    for dependency = requiredDependencies
        successCount = successCount + CheckIfInstalledAddOnsContainRequiredDependency(installedAddons, dependency);
    end
end

function isSuccess = CheckIfInstalledAddOnsContainRequiredDependency(installedAddOns, requiredDependency)
    fprintf("  Checking if Add-On ""%s"" is installed and enable...", requiredDependency);
    info = GetAddOnInformation(installedAddOns, requiredDependency);
    isSuccess = CheckIfAddOnIsInstalledAndEnabled(info, requiredDependency);
    if isSuccess
        disp('Yes!');
    end
end

function info = GetAddOnInformation(installedAddOns, addonName)
    info = installedAddOns(strcmp(installedAddOns.Name, addonName), :);
end

function isSuccess = CheckIfAddOnIsInstalledAndEnabled(info, addonName)
   if IsInstalled(info)
       isSuccess = CheckIfAddOnIsEnabled(info);
   else
       fprintf(2, " No%c    Add-On ""%s"" is not installed. Please install it.%c", newline, addonName, newline)
       isSuccess = 0;
   end
end

function isSuccess = CheckIfAddOnIsEnabled(info)
    if info.Enabled
        isSuccess = 1;
    else
       fprintf(2, " No%c    Add-On ""%s"" is installed but not enabled. Please enable it.%c",newline, info.Name, newline)
        isSuccess = 0;
    end
end

function isSuccess = IsInstalled(info)
    isSuccess = size(info, 1);
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