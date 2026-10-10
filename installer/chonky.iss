#define MyAppName "Chonky"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{1AC30AE6-FE99-4410-AD77-119AC1F9B245}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=is-leeroy-jenkins
DefaultDirName={localappdata}\Programs\Chonky
DefaultGroupName=Chonky
OutputDir=..\dist-installer
OutputBaseFilename=Chonky-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayName=Chonky
UninstallDisplayIcon={app}\Chonky.exe
CloseApplications=yes

[Files]
Source: "..\dist\Chonky\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Icons]
Name: "{group}\Chonky"; Filename: "{app}\Chonky.exe"
Name: "{autodesktop}\Chonky"; Filename: "{app}\Chonky.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Chonky.exe"; Description: "Launch Chonky"; Flags: nowait postinstall skipifsilent
