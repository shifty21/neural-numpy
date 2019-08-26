#include <stdio.h>
#include <omp.h>
#include <limits.h>

void foobar(char *arr, int n)
{
  int j;
  int temp=0;
  /* for (i = 0; i < m; i++) */
  /*   for (j = 0; j < n; j++) */
  /*     { */
  /*     printf("i == %d , j == %d, value == %d \n",i,j, *((arr+i*n) + j + temp)); */
  /*     temp++; */
  /*     } */
  for (j=0;j<n;j++){
    printf("j==%d value == %d \n",j,*(arr + j));
      temp++;
  }
}
extern "C"
{
  int matrix_multiply(char *arr1, char *arr2, int n){
    int j;
    short result=0;
    int temp = 0;
    /* printf("size of char %d",sizeof(arr1));  */
    for(j=0;j<n;j++){
      int r=0;
      char op1= *(arr1+j);
      char op2= *(arr2+j);
      if (op1!=0 && op2!=0){
        r=op1 * op2;
        result = result + r;
        // printf("counter=%d  x=%d and y=%d => prod=%d ===== result=%d\n",temp,(*(arr1+j)),(*(arr2+j)),r, result);
      }
      temp++;

    }
    /* printf("result=%d",result); */
    return result;
  }

}
int main()
{
   return 0;
}
