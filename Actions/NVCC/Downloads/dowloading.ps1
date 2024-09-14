

function SETUP_CUDART 
{
      echo "SETUP_CUDART ..."
      
      $url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-11.5.50-archive.zip"
      $output="cudart.zip"
      Invoke-WebRequest -Uri $url -OutFile $output
      Expand-Archive $output
      cd cudart\cuda_cudart-windows-x86_64-11.5.50-archive
      ls 

      cd ${HOME}
}

function SETUP_NVCC
{ 
      echo "SETUP_NVCC ..."
      
      $url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-11.5.50-archive.zip"
      $output="nvcc.zip"
      Invoke-WebRequest -Uri $url -OutFile $output
      Expand-Archive $output
      cd nvcc
      cd cuda_nvcc-windows-x86_64-11.5.50-archive
      cd bin
      .\nvcc.exe --version 
      [Environment]::SetEnvironmentVariable("NVCC","${PWD}\nvcc.exe")
      
      cd ${HOME}
}

$RELEASE=11.5.50

SETUP_CUDART
SETUP_NVCC

$env:NVCC
&$env:NVCC --version 



