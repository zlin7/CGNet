/* -----------------------------------------------------------------------------

Copyright Imre Risi Kondor, October 2006. Not for general release. 
Reproduction and distribution prohibited except when specifically 
authorized by the author. 

----------------------------------------------------------------------------- */

#ifndef _ClebschGordan
#define _ClebschGordan

#include <string>
#include <math.h>


using namespace std;

class ClebschGordan{
public:
  
  ClebschGordan(int _L); // loads from disk if available
  ClebschGordan(int _L, int dummy); // this one always computes from scratch
  ClebschGordan(char* filename);
  ~ClebschGordan();
  
  double operator()(int l1, int l2, int l, int m1, int m2, int m);

  void allocate();
  int computeSlow();
  int computeRecursively();
  int load(char* filename);
  int save(char* filename);

  void print();

  int L;
  int group;

private:

  double***** table;

  double slowCG(int l1, int l2, int l, int m1, int m2, int m);

};


#endif
