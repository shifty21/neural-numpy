#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include "behavioral.hpp"
#include <cstdio>
//#include <cstdint>
using namespace std;

long long int custom_multiplier(int arr1, int arr2) {
  //***insert size of multiplier here***
  int N = 8; //width
  int M = 8; //heigth
  bool a[N]; //width
  bool b[M]; //height
  bool pp[M/2][N+4]; //partial product
  bool p[N+M];
  bool acc_p[N+M]; //accurate product
  Lut Lut0[M/2][N+3];
  Lut Lut1[N+M];
  Lut Lut2[N+M];
  Carry Carry0[M/2][N+3];
  Carry Carry1[N+M-2];
  Carry Carry2[N+M];
  signed int aa, bb, saveaa, savebb, saveacc, error;
  signed int acc = 0;
  long long int ppp = 0;
  //bool row[M/2-1][(M/2-1)*2+N+1]; //is this correct?
  bool row[(M/2)+2][M+N];
  bool tmp[1][M+N];

  saveaa = (arr1);
  savebb = (arr2);
  // printf("saveaa -> %d, savebb -> %d\n",saveaa,savebb);
  //convert int to binary array

  aa = saveaa;
  bb = savebb;
  acc = aa*bb;
  saveacc = acc;

  for (int i=0; i < N; i++){
    a[i] = aa & 0x1;
    aa = aa >> 1;
  }

  for (int i=0; i < M; i++){
    b[i] = bb & 0x1;
    bb = bb >> 1;
  }

  for (int i=0; i < N+M; i++){
    acc_p[i] = acc & 0x1;
    acc = acc >> 1;
  }

//multiplier functionality

  for (int i = 0; i < M/2; i++){ //rows
    for (int j = 0; j < N+3; j++){ //columns
      if (i == 0){
        if (j == 0) {
          Lut0[i][j].setLutInput(a[1], a[0], 0, b[2*i], b[(2*i)+1], 1);
          pp[i][0] = Lut0[i][j].getA1O6();
          pp[i][1] = Lut0[i][j].getA1O5();
        }
        if (j == 1) {
          Lut0[i][j].setLutInput(a[2], a[1], a[0], 0, b[2*i], b[(2*i)+1]);
	  pp[i][2] = Lut0[i][j].getA2O6();
        }
	if (j == 2) {
	  Lut0[i][j].setLutInput(a[2], a[1], a[0], 0, b[2*i], b[(2*i)+1]);
	  Carry0[i][j].setCarryInput(1, 0, Lut0[i][j].getCGO6());
	}
        if ((j > 2) && (j < N)){
          Lut0[i][j].setLutInput(a[j-1], a[j], 0, b[2*i], b[(2*i)+1], 0);
          //Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), a[j], Lut0[i][j].getAO6());
	  Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N){
          Lut0[i][j].setLutInput(a[N-1], a[N-1], 0, b[2*i], b[(2*i)+1], 0);
          //Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), a[N-1], Lut0[i][j].getAO6());
	  Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+1){
          Lut0[i][j].setLutInput(1, a[N-1], 0, b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 1, Lut0[i][j].getBO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+2){
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 1, 1);
	  pp[i][j] = Carry0[i][j].getCarryXOut();
	  pp[i][j+1] = Carry0[i][j].getCarryCOut();
        }
      }
      if ((i > 0) && (i < M/2-1)){
        if (j == 0){
          Lut0[i][j].setLutInput(a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1], 1);
          pp[i][0] = Lut0[i][j].getA1O6();
          pp[i][1] = Lut0[i][j].getA1O5();
        }
        if (j == 1) {
          Lut0[i][j].setLutInput(a[2], a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1]);
	  pp[i][2] = Lut0[i][j].getA2O6();
        }
	if (j == 2) {
	  Lut0[i][j].setLutInput(a[2], a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1]);
	  Carry0[i][j].setCarryInput(1, 0, Lut0[i][j].getCGO6());
	}
        if ((j > 2) && (j < N)){
          Lut0[i][j].setLutInput(a[j-1], a[j], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N){
          Lut0[i][j].setLutInput(a[N-1], a[N-1], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+1){
          Lut0[i][j].setLutInput(0, a[N-1], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getBO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+2){
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 1, 1);
	  pp[i][j] = Carry0[i][j].getCarryXOut();
	  pp[i][j+1] = Carry0[i][j].getCarryCOut();
        }
      }
      if (i == M/2-1){
        if (j == 0){
          Lut0[i][j].setLutInput(a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1], 1);
          pp[i][0] = Lut0[i][j].getA1O6();
          pp[i][1] = Lut0[i][j].getA1O5();
        }
        if (j == 1) {
          Lut0[i][j].setLutInput(a[2], a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1]);
	  pp[i][2] = Lut0[i][j].getA2O6();
        }
	if (j == 2) {
	  Lut0[i][j].setLutInput(a[2], a[1], a[0], b[(2*i)-1], b[2*i], b[(2*i)+1]);
	  Carry0[i][j].setCarryInput(1, 0, Lut0[i][j].getCGO6());
	}
        if ((j > 2) && (j < N)){
          Lut0[i][j].setLutInput(a[j-1], a[j], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N){
          Lut0[i][j].setLutInput(a[N-1], a[N-1], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getAO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+1){
          Lut0[i][j].setLutInput(0, a[N-1], b[(2*i)-1], b[2*i], b[(2*i)+1], 0);
          Carry0[i][j].setCarryInput(Carry0[i][j-1].getCarryCOut(), 0, Lut0[i][j].getBO6());
	  pp[i][j] = Carry0[i][j].getCarryXOut();
        }
        if (j == N+2){
	  pp[i][j] = 0;
	  pp[i][j+1] = 0;
        }
      }
    }
  }

for (int i = 0; i < M/2; i++) {
  for (int j = 0; j < N+M; j++) {
    if (i == 0) {
      if (j < N+4){
        row[i][j] = pp[i][j];
      }
      if (j >= N+4){
        row[i][j] = 0;
      }
    }
    if (i != 0 and i != M/2-1) {
      if (j < i*2) {
        row[i][j] = 0;
      }
      if (j >= i*2 and j < ((i*2)+N+4)) {
        row[i][j] = pp[i][j-(i*2)];
      }
      if (j >= ((i*2)+N+4)) {
        row[i][j] = 0;
      }
    }
    if (i == M/2-1) {
      if (j < i*2) {
        row[i][j] = 0;
      }
      if (j >= i*2 and i < ((i*2)+N+4)) {
        row[i][j] = pp[i][j-(i*2)];
      }
    }
  }
}

//------------------------------------------------------------------------------
//8-Bit
//------------------------------------------------------------------------------

p[0] = pp[0][0];
p[1] = pp[0][1];

for (int i = 2; i < N+M; i++) {
  if (i == 2) {
    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], row[2][13], 0);
    Carry1[i].setCarryInput(0, Lut1[i].getCompO5_mix(), Lut1[i].getCompO6_mix());
    p[i] = Carry1[i].getCarryXOut();
  }
  if (i != 2 and i != 12 and i != 13) {
    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], row[2][13], 0);
    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_mix(), Lut1[i].getCompO6_mix());
    p[i] = Carry1[i].getCarryXOut();
  }
  if (i == 12) {
    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], row[2][13], 0);
    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_mix_12(), Lut1[i].getCompO6_mix_12());
    p[i] = Carry1[i].getCarryXOut();
  }
  if (i == 13) {
    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], row[2][13], 0);
    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_mix_13(), Lut1[i].getCompO6_mix_13());
    p[i] = Carry1[i].getCarryXOut();
  }
}

ppp = 0;
for (int i = 0; i < N+M; i++){
  ppp += p[i]*(pow(2,i));
}

if (p[N+M-1]){
  long long int tmp;
  tmp = pow(2, N+M);
  tmp = tmp - 1;
  ppp = ~ppp & tmp;
  ppp = (ppp + 1) * -1;
}
 return ppp;
}

extern "C"
{
  // int8_t matrix_multiply(char *arr1, char *arr2, int n, int stride_arr1, int stride_arr2){

  int16_t matrix_multiply(char *arr1, char *arr2, int n){
  int j;
    short result=0;
    int temp = 0;
    // printf("stride length of arr1=%d and stride length of arr2=%d\n",stride_arr1, stride_arr2);
    for(j=0;j<n;j++){
      short r=0;
      short op1= (short)(*(arr1+j));
      short op2= (short)(*(arr2+j));
      if (op1!=0 && op2!=0){
        // if (op2 < 0 && op1 >0){
        //     r= custom_multiplier(op2,op1);
        // } else if (op2 <0 && op1 <0) {
        //   if (abs(op2)> abs(op1)) {
        //     r = custom_multiplier(op2,op1);
        //   } else {
        //     r = custom_multiplier(op1,op2);
        //   }
        // }
        // else {
        //   r = custom_multiplier(op1,op2);
        // }
        r = op1*op2;
        r = r>>2;
        // if (r < -128){
        //   r = -128;
        //     } else if (r > 127) {
        //   r = 127;
        // }
         result = result + r;
        // printf("counter=%d  op1=%d and op2=%d => prod=%d ===== result=%d\n",temp, op1,op2,r, result);
      }
      temp++;

    }
    // printf("result=%d\n",result);
    return result;
  }

}
int main(int argc, char **argv) {
  return 0;
}
