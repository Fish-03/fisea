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
