
function CUDART_SETUP 
{
      echo "CUDART_SETUP ..."
      
      ## 1.0 
      $url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-11.5.50-archive.zip"
      $output="cudart.zip"
      
      Invoke-WebRequest -Uri $url -OutFile $output
      Expand-Archive $output -DestinationPath .   
      
      $CUDART="cuda_cudart-windows-x86_64-11.5.50-archive"
      
      
      ## 2.0 
      Set-Location -Path $CUDART
      Get-ChildItem 
      
      $env:CUDART_PATH="$ROOT\$CUDART"
      $env:CUDART_LIB="$env:CUDART_PATH\lib"
      $env:CUDART_INC="$env:CUDART_PATH\include"      

      $env:CUDART_PATH=("$env:CUDART_PATH").replace("\","\\")
      $env:CUDART_LIB=("$env:CUDART_LIB").replace("\","\\")
      $env:CUDART_INC=("$env:CUDART_INC").replace("\","\\")
      
      Set-Location -Path $ROOT
}


function NVCC_SETUP
{ 
      echo "NVCC_SETUP ..."

      ## 1.0 
      $url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-11.5.50-archive.zip"
      $output="nvcc.zip"
      
      Invoke-WebRequest -Uri $url -OutFile $output
      Expand-Archive $output -DestinationPath .   
      
      $NVCC="cuda_nvcc-windows-x86_64-11.5.50-archive"

      ## 2.0 
      Set-Location -Path $NVCC 
      Set-Location bin 
      Get-ChildItem 
      Get-ChildItem nvcc.exe 
 
      $env:NVCC_PATH="$ROOT\$NVCC\bin" 
      $env:NVCC="$env:NVCC_PATH\nvcc.exe"  
      
      $env:NVCC_PATH=("$env:NVCC_PATH").replace("\","\\")
      $env:NVCC=("$env:NVCC").replace("\","\\")
 
      Set-Location -Path $ROOT
}

function CL_SETUP
{
  echo "CL_SETUP ..."
   
  $VSWHERE="C:\ProgramData\Chocolatey\bin\vswhere.exe"

  $VSTOOLS = &($VSWHERE) -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  Write-Output "[VSTOOLS]:'$VSTOOLS' "

  if($VSTOOLS) 
  {
    $VSTOOLS = join-path $VSTOOLS 'Common7\Tools\vsdevcmd.bat'
    if (test-path $VSTOOLS) 
    {
      cmd /s /c " ""$VSTOOLS""  -arch=x64 -host_arch=x64 $args && set" | where { $_ -match '(\w+)=(.*)' } | 
      foreach{$null = new-item -force -path "Env:\$($Matches[1])" -value $Matches[2] }
    }
  }
  
  cl.exe 
  cmake.exe --version 
  ninja.exe --version 
}


function COMPILATION_OBVIOUS 
{ 
      echo "COMPILATION_OBVIOUS ..."
      &$env:NVCC -o simplest.exe ../simplest.cu -I"$env:CUDART_INC" -L"$env:CUDART_LIB"
      .\simplest.exe
      rm simplest.exe 
}


function COMPILATION_NICEST 
{ 
      echo "COMPILATION_NICEST ..."
      $env:INCLUDES=" -I ${env:CUDART_INC}"
      $env:LIBRARIES=" -LIBPATH:${env:CUDART_LIB} "
      
      &$env:NVCC -o simplest.exe ../simplest.cu 
      .\simplest.exe
      rm simplest.exe 
      
      $env:PATH="$env:PATH;$env:NVCC_PATH;"
      nvcc.exe --version 
      
      nvcc.exe -o simplest.exe ../simplest.cu
      .\simplest.exe
      rm simplest.exe       
}


function COMPILATION_PERFECT 
{
      echo "COMPILATION_PERFECT ..."
      cmake.exe .. -G Ninja 
      ninja.exe
      ctest.exe
}


## 0. 
$ROOT=Get-Location
$RELEASE=11.5.50


## 1. SETUP ...
NVCC_SETUP
&$env:NVCC --version 

CUDART_SETUP
$env:CUDART_LIB
$env:CUDART_INC

CL_SETUP

## 2. RUNNING ...
New-Item -ItemType Directory BUILD
Set-Location BUILD


COMPILATION_OBVIOUS
COMPILATION_NICEST 
COMPILATION_PERFECT
#COMPILATION_SIMPLEST




#$env:CUDA_PATH
#$env:CUDACXX
#&$env:CUDACXX --version 


## 2.C. FANCIEST (https://www.collinsdictionary.com/dictionary/english/fanciest) 
#
#$CMAKE_CUDA_COMPILER=($env:CUDACXX).replace("\","\\")
#$CUDAToolkit_ROOT=($env:CUDA_PATH).replace("\","\\")
#
#echo $CMAKE_CUDA_COMPILER
#cmake.exe .. -G Ninja -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT"
#
#-DCMAKE_CUDA_ARCHITECTURES="all"
#ninja.exe
#ctest.exe
#
