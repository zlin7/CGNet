/* -----------------------------------------------------------------------------

Copyright Imre Risi Kondor, October 2006. Not for general release. 
Reproduction and distribution prohibited except when specifically 
authorized by the author. 

----------------------------------------------------------------------------- */

#include <iostream>
#include <stdio.h>
#include <fcntl.h>

#include "ClebschGordan.hpp"

#ifndef MIN 
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y)) 
#endif

#ifndef MAX 
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y)) 
#endif

using namespace std;

main(int argc, char** argv){
  
  if(argc!=2){
    cout<<endl;
    cout<<"  precomputeCG : computes and saves the Clebsch-Gordan coefficients of SO(3) to file"<<endl;
    cout<<endl;
    cout<<"  Usage: precomputeCG <L>"<<endl<<endl;
    cout<<"  L   maximum l value up to which CG coefficients are computed"<<endl<<endl;
  }

  int L; sscanf(argv[1],"%d",&L);
  
  //cout<<"Computing ..."<<endl;

  ClebschGordan CG(L,0);
  //CG.print();

  double****** Tl1=new double*****[L+1];
  double****** table=Tl1;
  for(int l1=0; l1<=L; l1++){
    double***** Tl2=new double****[l1+1];
    Tl1[l1]=Tl2;
    for(int l2=0; l2<=l1; l2++){
      int loffset=l1-l2;
      double**** Tl=new double***[MIN(l1+l2,L)-loffset+1];
      Tl2[l2]=Tl;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
        //cout<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl;
        double*** Tm=new double**[2*l2+1];
        
        Tl[l-loffset]=Tm;
        for(int m = -l2; m <= l2; m++){
          double** Tm1=new double*[2*l1+1];
          Tm[m+l2]=Tm1;
            for(int m1=-l1; m1<=l1; m1++){
              double* Tm2=new double[2*l2+1];
              Tm1[m1+l1]=Tm2;
              for(int m2=-l2; m2<=l2;m2++){
                Tm2[m2+l2]=CG(l1,l2,l,m1,m2,m);
                cout << Tm2[m2+l2] << ",";
              }
              cout << endl;
          }
        }
      }
    }
    cout<<"."; cout.flush();
  }
  printf("%f\n", CG(0,0,0,0,0,0));


  /*
  char filename[255];
  sprintf(filename,"ClebschGordan_SO3_L%d.cg",L);
  cout<<"Saving to file "<<filename<<" ... "; cout.flush();

  if(CG.save(filename)==0) cout<<"done."<<endl;

  */
  //ClebschGordan CG2(filename);
  //CG2.print();

}
