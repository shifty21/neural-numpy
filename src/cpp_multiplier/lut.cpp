#include <iostream>
#include "math.h"
#include <cmath>
#include "behavioral.hpp"
using namespace std;

void Lut::setLutInput( bool in0, bool in1, bool in2, bool in3, bool in4, bool in5 ) {
  i0 = in0; //d
  i1 = in1; //c
  i2 = in2; //b
  i3 = in3; //a
  i4 = in4; //switch
  i5 = in5; //static '1'
}

//----------------mix----------------

bool Lut::getCompO6_mix( void ) {
  bool out [32] = {0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix( void ) {
  bool out [32] = {0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_mix_12( void ) {
  bool out [32] = {0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_12( void ) {
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_mix_13( void ) {
  bool out [32] = {1,0,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_13( void ) {
  bool out [32] = {0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

//----------------sum----------------

bool Lut::getCompO6_sum( void ){
  bool out [32] = {0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_sum( void ){
  bool out [32] = {0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_sum_12( void ){
  bool out [32] = {0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_sum_12( void ){
  bool out [32] = {0,0,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_sum_13( void ){
  bool out [32] = {1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_sum_13( void ){
  bool out [32] = {0,1,1,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

//----------------carry----------------

bool Lut::getCompO6_carry( void ){
  bool out [32] = {0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_carry( void ){
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_carry_11( void ){
  bool out [32] = {0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_carry_11( void ){
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_carry_12( void ){
  bool out [32] = {0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_carry_12( void ){
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_carry_13( void ){
  bool out [32] = {1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_carry_13( void ){
  bool out [32] = {1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

//----------------mix_rev----------------

bool Lut::getCompO6_mix_rev( void ) {
  bool out [32] = {0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_rev( void ) {
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_mix_rev_11( void ) {
  bool out [32] = {0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_rev_11( void ) {
  bool out [32] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_mix_rev_12( void ) {
  bool out [32] = {0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_rev_12( void ) {
  bool out [32] = {0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getCompO6_mix_rev_13( void ) {
  bool out [32] = {1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getCompO5_mix_rev_13( void ) {
  bool out [32] = {0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

//------------------------------------
//----------------rest----------------
//------------------------------------

bool Lut::getA1O6( void ) {
  bool out [32] = {0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }
bool Lut::getA1O5( void ) {
  bool out [32] = {0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getA2O6( void ) {
  bool out [64] = {0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4)) + ((int)i5 * pow(2, 5));
  return out[tmp];
  }

bool Lut::getCGO6( void ) {
  bool out [64] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4)) + ((int)i5 * pow(2, 5));
  return out[tmp];
  }

bool Lut::getAO6( void ) {
  bool out [32] = {0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
  }

bool Lut::getBO6( void ) {
  bool out [32] = {1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1)) + ((int)i2 * pow(2, 2)) + ((int)i3 * pow(2, 3)) + ((int)i4 * pow(2, 4));
  return out[tmp];
}

bool Lut::getRcaO6( void ) {
  bool out [4] = {0,1,1,0};
  int tmp = ((int)i0 * pow(2, 0)) + ((int)i1 * pow(2, 1));
  return out[tmp];
}
