$VSWHERE="C:\ProgramData\Chocolatey\bin\vswhere.exe"

$VSTOOLS = &($VSWHERE) -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Write-Output "[VSTOOLS]:'$VSTOOLS' "

if($VSTOOLS) 
{
  $VSTOOLS = join-path $VSTOOLS 'Common7\Tools\vsdevcmd.bat'
  if (test-path $VSTOOLS) 
  {
    cmd /s /c """$VSTOOLS"" $args && set" | where { $_ -match '(\w+)=(.*)' } | 
    foreach{$null = new-item -force -path "Env:\$($Matches[1])" -value $Matches[2] }
  }
}
           
cl.exe 

New-Item -ItemType Directory BUILD
Set-Location BUILD

cmake.exe .. -G Ninja 

ninja.exe 

ctest.exe

Get-ChildItem 

.\simplest.exe 
