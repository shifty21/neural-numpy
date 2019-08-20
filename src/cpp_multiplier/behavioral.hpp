//Lut
class Lut {
public:
  bool i0, i1, i2, i3, i4, i5; //inputs
  bool o5, o6; //output

  void setLutInput( bool in0, bool in1, bool in2, bool in3, bool in4, bool in5 );
  bool getCompO6_mix( void );
  bool getCompO5_mix( void );
  bool getCompO6_mix_12( void );
  bool getCompO5_mix_12( void );
  bool getCompO6_mix_13( void );
  bool getCompO5_mix_13( void );
  bool getCompO6_mix_rev( void );
  bool getCompO5_mix_rev( void );
  bool getCompO6_mix_rev_11( void );
  bool getCompO5_mix_rev_11( void );
  bool getCompO6_mix_rev_12( void );
  bool getCompO5_mix_rev_12( void );
  bool getCompO6_mix_rev_13( void );
  bool getCompO5_mix_rev_13( void );
  bool getCompO6_sum( void );
  bool getCompO5_sum( void );
  bool getCompO6_sum_12( void );
  bool getCompO5_sum_12( void );
  bool getCompO6_sum_13( void );
  bool getCompO5_sum_13( void );
  bool getCompO6_carry( void );
  bool getCompO5_carry( void );
  bool getCompO6_carry_11( void );
  bool getCompO5_carry_11( void );
  bool getCompO6_carry_12( void );
  bool getCompO5_carry_12( void );
  bool getCompO6_carry_13( void );
  bool getCompO5_carry_13( void );
  bool getA1O6( void );
  bool getA1O5( void );
  bool getA2O6( void );
  bool getCGO6( void );
  bool getAO6( void );
  bool getBO6( void );
  bool getRcaO6( void );
};

//carry chain element (single!!!)
class Carry {
public:
  bool inc, ino5, ino6;
  bool xout, cout;

  void setCarryInput( bool incin, bool ino5in, bool ino6in );
  bool getCarryCOut( void );
  bool getCarryXOut( void );
};