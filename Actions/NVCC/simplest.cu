#include <stdio.h>

__global__
void cuda_hello()
{
  printf(">> [smallest] Hello from GPU!\n");
}

int main()
{
    printf(">> [smallest] Hello from CPU!\n");   
    cuda_hello<<<1,1>>>();
    return 0;
}
