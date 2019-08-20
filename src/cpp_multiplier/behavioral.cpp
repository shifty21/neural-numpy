#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <cmath>
#include <fstream>
#include "behavioral.hpp"
//#include <cstdint>
using namespace std;

int main(int argc, char **argv) {
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

  if (argc != 3){
    std::cout << "Wrong number of inputs" << endl;
    return -1;
  }

  saveaa = atoi(argv[1]);
  savebb = atoi(argv[2]);

  //uncomment this for console input from here -->
  //std::cout << "Input a: " << endl;
  //std::cin >> saveaa;
  //std::cout << "Input b: " << endl;
  //std::cin >> savebb;
  //<-- till here

  //uncomment this for reading from 2 .txt files and write into a .csv from here -->
  //std::ofstream ofile;
  //ofile.open("mul_results.csv", ios::out | ios::trunc);

  //ofile << "a,b,bin_a,bin_b,accu,appr,bin_accu,bin_appr,row0,row1,row2,row3,error,rel_err" << endl;

  //ofile.close();

  //std::ifstream infilea("tb_in_a0.txt");
  //while (infilea >> saveaa){
  //std::ifstream infileb("tb_in_b0.txt");
  //while (infileb >> savebb){
  //<-- till here

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
//10x24-Bit for video gray scale mul
//------------------------------------------------------------------------------

//for (int i = 0; i < N+M; i++) {
//  if (i == 0) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 0, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5_mix_rev(), Lut1[i].getCompO6_mix());
//    row[5][i] = Carry1[i].getCarryXOut();
//  }
//  if (i != 0 ){//and i != 11 and i != 12 and i != 13) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 0, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_mix(), Lut1[i].getCompO6_mix());
//    row[5][i] = Carry1[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[5][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, row[5][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(row[5][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), row[5][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//}

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

//------------------------------------------------------------------------------
//16-Bit
//------------------------------------------------------------------------------

//for (int i = 0; i < N+M; i++) {
//  if (i == 0) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 0, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5_carry(), Lut1[i].getCompO6_carry());
//    row[8][i] = Carry1[i].getCarryXOut();
//  }
//  if (i != 0 ){
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 0, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_carry(), Lut1[i].getCompO6_carry());
//    row[8][i] = Carry1[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++) {
//  if (i == 0) {
//    Lut1[i].setLutInput(row[7][i], row[6][i], row[5][i], row[4][i], 0, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5_carry(), Lut1[i].getCompO6_carry());
//    row[9][i] = Carry1[i].getCarryXOut();
//  }
//  if (i != 0 ){
//    Lut1[i].setLutInput(row[7][i], row[6][i], row[5][i], row[4][i], 0, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5_carry(), Lut1[i].getCompO6_carry());
//    row[9][i] = Carry1[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[8][i], row[9][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, row[9][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(row[8][i], row[9][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), row[9][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit
//------------------------------------------------------------------------------
//10-Bit_1

//tmp[1][0] = pp[0][0];
//tmp[1][1] = pp[0][1];

//for (int i = 2; i < N+M; i++){
//  if (i == 2) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 2) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(tmp[1][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, tmp[1][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(tmp[1][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), tmp[1][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit_2

//tmp[1][0] = pp[1][0];
//tmp[1][1] = pp[1][1];

//for (int i = 2; i < N+M; i++){
//  if (i == 2) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], row[2][i], row[1][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 2) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], row[2][i], row[1][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[0][i], tmp[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, tmp[1][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(row[0][i], tmp[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), tmp[1][i], Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit_3

//tmp[1][0] = pp[0][0];
//tmp[1][1] = pp[0][1];

//for (int i = 2; i < N+M; i++) {
//  if ( i == 2) {
//    Lut2[i].setLutInput(row[0][i], row[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, row[0][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 2) {
//    Lut2[i].setLutInput(row[0][i], row[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), row[0][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++){
//  if (i == 0) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], row[2][i], tmp[1][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], row[2][i], tmp[1][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit_4

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[3][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(row[3][i], row[4][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++){
//  if (i == 0) {
//    Lut1[i].setLutInput(tmp[1][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut1[i].setLutInput(tmp[1][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit_5

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[1][i], row[2][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut2[i].setLutInput(row[1][i], row[2][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++){
//  if (i == 0) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], tmp[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut1[i].setLutInput(row[4][i], row[3][i], tmp[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//10-Bit_6

//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(row[2][i], row[3][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//  if (i > 0) {
//   Lut2[i].setLutInput(row[2][i], row[3][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), tmp[1][i], Lut2[i].getRcaO6());
//    tmp[1][i] = Carry2[i].getCarryXOut();
//  }
//}

//for (int i = 0; i < N+M; i++){
//  if (i == 0) {
//    Lut1[i].setLutInput(row[4][i], tmp[1][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//  if (i > 0) {
//    Lut1[i].setLutInput(row[4][i], tmp[1][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    p[i] = Carry1[i].getCarryXOut();
//  }
//}

//------------------------------------------------------------------------------
//16-Bit
//------------------------------------------------------------------------------

//tmp[0][0] = pp[0][0];
//tmp[0][1] = pp[0][1];

//for (int i = 2; i < N+M; i++){
//  if (i == 2) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[0][i] = Carry1[i].getCarryXOut();
//  }
//  if (i != 2) {
//    Lut1[i].setLutInput(row[3][i], row[2][i], row[1][i], row[0][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[0][i] = Carry1[i].getCarryXOut();
//  }
//}
//
//for (int i = 0; i < N+M; i++){
//  if (i == 0) {
//    Lut1[i].setLutInput(row[7][i], row[6][i], row[5][i], row[4][i], 1, 0);
//    Carry1[i].setCarryInput(0, Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//  if (i != 0) {
//    Lut1[i].setLutInput(row[7][i], row[6][i], row[5][i], row[4][i], 1, 0);
//    Carry1[i].setCarryInput(Carry1[i-1].getCarryCOut(), Lut1[i].getCompO5(), Lut1[i].getCompO6());
//    tmp[1][i] = Carry1[i].getCarryXOut();
//  }
//}
//
//for (int i = 0; i < N+M; i++) {
//  if ( i == 0) {
//    Lut2[i].setLutInput(tmp[0][i], tmp[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(0, 0, Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//  if (i != 0) {
//    Lut2[i].setLutInput(tmp[0][i], tmp[1][i], 0, 0, 0, 0);
//    Carry2[i].setCarryInput(Carry2[i-1].getCarryCOut(), 0, Lut2[i].getRcaO6());
//    p[i] = Carry2[i].getCarryXOut();
//  }
//}

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

//regular console output, uncomment from here -->
//std::cout << "Result: ";
std::cout << ppp << endl;
//<-- till here

//give .csv conform output, uncomment from here -->

//ofile.open("mul_results.csv", ios::out | ios::app);

//ofile << saveaa << "," << savebb << ",";
//for (int i = N-1; i >= 0; i--){
//  ofile << a[i];
//}
//ofile << ",";
//for (int i = M-1; i >= 0; i--){
//  ofile << b[i];
//}
//ofile << ",";
//ofile << saveacc << "," << ppp << ",";
//for (int i = N+M-1; i >= 0; i--){
//  ofile << acc_p[i];
//}
//ofile << ",";
//for (int i = N+M-1; i >= 0; i--){
//  ofile << p[i];
//}

//>>>
//ofile << ",";
//for (int j = 0; j < M/2; j++) {
//  ofile << ",";
//  for (int i = N+M-1; i >= 0; i--){
//    ofile << row[j][i];
//  }
//}
//ofile << ",";
//for (int i = N+M-1; i >= 0; i--){
//  ofile << row[1][i];
//}
//ofile << ",";
//for (int i = N+M-1; i >= 0; i--){
//  ofile << row[2][i];
//}
//ofile << ",";
//for (int i = N+M-1; i >= 0; i--){
//  ofile << row[3][i];
//}
///<<<

//ofile << endl;
//ofile.close();
//}
//}
//<-- till here (sorry for the bracket layout of the output :D)
return 0;
}
