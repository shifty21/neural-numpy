#include <iostream>
#include "behavioral.hpp"
using namespace std;

void Carry::setCarryInput( bool incin, bool ino5in, bool ino6in ) {
  inc = incin;
  ino5 = ino5in;
  ino6 = ino6in;
}

bool Carry::getCarryCOut( void ) {
  cout = (ino5 & !ino6) | (inc & ino6);
  return cout;
}

bool Carry::getCarryXOut( void ) {
  xout = ino6 ^ inc;
  return xout;
}
