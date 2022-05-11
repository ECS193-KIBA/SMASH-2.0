function CheckAddOnsRequiredForSMASH
    disp("Checking Add-Ons Required for SMASH...");
    installedAddons = matlab.addons.installedAddons;
    requiredDependencies = SmashDependencies();
    successCount = CheckIfInstalledAddOnsContainAllRequiredDependencies(installedAddons, requiredDependencies);
    printSeparator();
    if successCount == size(requiredDependencies, 2)
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
    fprintf("  Checking if Add-On ''%s'' is installed and enable...", requiredDependency);
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
       fprintf(2, " No%c    Add-On ''%s'' is not installed. Please install it.%c", newline, addonName, newline)
       isSuccess = 0;
   end
end

function isSuccess = CheckIfAddOnIsEnabled(info)
    if info.Enabled
        isSuccess = 1;
    else
       fprintf(2, " No%c    Add-On ''%s'' is installed but not enabled. Please enable it.%c",newline, info.Name, newline)
        isSuccess = 0;
    end
end

function isSuccess = IsInstalled(info)
    isSuccess = size(info, 1);
end

function printSeparator
    disp(repmat('-',1,80))
end