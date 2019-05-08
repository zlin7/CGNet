/* -----------------------------------------------------------------------------

Copyright Imre Risi Kondor, October 2006. Not for general release. 
Reproduction and distribution prohibited except when specifically 
authorized by the author. 

----------------------------------------------------------------------------- */


#include "ClebschGordan.hpp"

#ifndef MIN 
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y)) 
#endif

#ifndef MAX 
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y)) 
#endif

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define CG_MAGIC 302432159
#define CG_VERSION 1

using namespace std;


inline int max( int i, int j ){return (i>j) ? i : j;}
inline int min( int i, int j ){return (i<j) ? i : j;}


ClebschGordan::ClebschGordan(int _L): L(_L){
  group=0;
  allocate();
  char filename[255];
  sprintf(filename,"ClebschGordan_SO3_L%d.cg",L);
  FILE* file=fopen(filename,"r");
  bool loaded=0;
  if(file!=NULL){
    fclose(file);
    loaded=load(filename);
  }
  if(!loaded){ 
    computeSlow();
    save(filename);
  }
}


ClebschGordan::ClebschGordan(int _L, int method): L(_L){
  group=0;
  allocate();
  switch(method){
  case 0:
    computeSlow();
    break;
  case 1:
    computeRecursively();
    break;
  }
}


void ClebschGordan::allocate(){
  double***** Tl1=new double****[L+1];
  table=Tl1;
  for(int l1=0; l1<=L; l1++){
    double**** Tl2=new double***[l1+1];
    Tl1[l1]=Tl2;
    for(int l2=0; l2<=l1; l2++){
      int loffset=l1-l2;
      double*** Tl=new double**[MIN(l1+l2,L)-loffset+1];
      Tl2[l2]=Tl;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
        //cout<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl;
        double** Tm1=new double*[2*l1+1];
        Tl[l-loffset]=Tm1;
        for(int m1=-l1; m1<=l1; m1++){
          int m2offset=MAX(-l2,-l-m1);
          double*Tm2=new double[MIN(l2,l-m1)-m2offset+1];
          Tm1[m1+l1]=Tm2;
        }
      }
    }
    cout<<"."; cout.flush();
  }
  cout<<endl;
}


int ClebschGordan::computeSlow(){

  for(int l1=0; l1<=L; l1++){
    double**** Tl2=table[l1];
    for(int l2=0; l2<=l1; l2++){
      double*** Tl=Tl2[l2];
      int loffset=l1-l2;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
        //cout<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl;
        double** Tm1=Tl[l-loffset];
        for(int m1=-l1; m1<=l1; m1++){
          int m2offset=MAX(-l2,-l-m1);
          double* Tm2=Tm1[m1+l1];
          for(int m2=m2offset; m2<=MIN(l2,l-m1); m2++){
            Tm2[m2-m2offset]=slowCG(l1,l2,l,m1,m2,m1+m2);
          }
        }
      }
    }
    cout<<"."; cout.flush();
  }
  cout<<endl;

  return 1;
}


int ClebschGordan::computeRecursively(){

  for(int l1=0; l1<=L; l1++){
    double**** Tl2=table[l1];
    for(int l2=0; l2<=l1; l2++){
      double*** Tl=Tl2[l2];
      int loffset=l1-l2;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
	double** Tm1=Tl[l-loffset];

	double* c; c=new double[2*l1+2];
	double* nc; nc=new double[2*l1+2];
	//	double c[2*l1+2];
	// double nc[2*l1+2];
	
	if(1){
	  int m=l;
	  for(int m1=-l1; m1<=l1; m1++){
	    int m2=m-m1;
	    if( m2>=-l2 && m2<=l2 ){
	      c[m1+l1]=slowCG(l1,l2,l,m1,m2,m);
	      Tm1[m1+l1][m2-MAX(-l2,-l-m1)]=c[m1+l1];
	    }
	    else c[m1+l1]=0;
	  }
	  c[2*l1+1]=0; // fictitious 
	  nc[2*l1+1]=0;  
	}

	
	for(int m=l-1; m>=-l; m--){
	  for(int m1=-l1; m1<=l1; m1++){
	    int m2=m-m1;
	    if( m2>=-l2 && m2<=l2 ){
	      // c[m1+l1]=slowCG(l1,l2,l,m1,m2,m);
	      nc[m1+l1]=(c[m1+l1+1]*sqrt((double)(l1+m1+1)*(l1-m1))+c[m1+l1]*sqrt((double)(l2+m2+1)*(l2-m2)))/(sqrt((double)(l+m+1)*(l-m)));
	      Tm1[m1+l1][m2-MAX(-l2,-l-m1)]=nc[m1+l1];
	      // if(abs(nc[m1+l1]-slowCG(l1,l2,l,m1,m2,m))>0.001) cout<<l1<<" "<<l2<<" "<<l<<" "<<m1<<" "<<m2<<" "<<m<<endl;
	    }
	    else nc[m1+l1]=0;
	  }
	  double* t=c; c=nc; nc=t;
	}

	//delete[] c;
	//delete[] nc;
      }
    }
    cout<<"."; cout.flush();
  }
  cout<<endl;

}


ClebschGordan::~ClebschGordan(){
  for(int l1=0; l1<=L; l1++){
    for(int l2=0; l2<=l1; l2++){
      int loffset=l1-l2;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
	for(int m1=-l1; m1<=l1; m1++){
	  delete[] table[l1][l2][l-loffset][m1+l1];
	}
	delete[] table[l1][l2][l-loffset]; 
      }
      delete[] table[l1][l2];
    }
    delete[] table[l1];
  }
  delete[] table;
}


double ClebschGordan::operator()(int l1, int l2, int l, int m1, int m2, int m){

  double inverter=1;

  if( l1<0 || l2<0 || l<0 ) return 0;
  if( l1>L || l2>L || l>L ) return 0;
  
  if( m1<-l1 || m1>l1 ) return 0;
  if( m2<-l2 || m2>l2 ) return 0;
  if( m<-l || m>l ) return 0;

  if(l2>l1){
    int t;
    t=l1; l1=l2; l2=t;
    t=m1; m1=m2; m2=t;
    if((l1+l2-l)%2==1) inverter=-1;
  }

  if(l<l1-l2) return 0;
  
  if(m!=m1+m2) return 0;
  
  if(m2<0){
    m1=-m1;
    m2=-m2;
    if((l1+l2-l)%2==1) inverter*=-1;
  }

  int loffset=l1-l2;
  int m2offset=MAX(-l2,-l-m1);
  
  //cout<<l1<<","<<l2<<","<<l-loffset<<","<<m1+l1<<","<<m2-m2offset<<endl;
  
  return table[l1][l2][l-loffset][m1+l1][m2-m2offset]*inverter;

}


int ClebschGordan::load(char* filename){
  FILE* file=fopen(filename,"r");
  if(file==NULL) {cout<<" Error: Cannot open file "<<file<<endl; return -1;}
  int readlength;

  double magic;
  readlength=fread(&magic,sizeof(double),1,file);
  if(readlength!=1){cout<<" Error in reading from file "<<file<<endl; fclose(file); return -1;}
  if(magic!=CG_MAGIC){cout<<"Magic number doesn't match in file "<<file<<endl; fclose(file); return -1;}

  int version;
  readlength=fread(&version,sizeof(int),1,file);
  if(readlength!=1){cout<<" Error in reading from file "<<file<<endl; fclose(file); return -1;}
  if(version!=CG_VERSION){cout<<" File format version number does not match in "<<file<<endl; fclose(file); return -1;}

  readlength=fread(&group,sizeof(int),1,file);
  if(readlength!=1){cout<<" Error in reading from file "<<file<<endl; fclose(file); return -1;}
  if(group!=0){cout<<"Unrecognized group identified in file "<<file<<endl; fclose(file); return -1;}

  readlength=fread(&L,sizeof(int),1,file);
  if(readlength!=1){cout<<" Error in reading from file "<<file<<endl; fclose(file); return -1;}

  int dsize=sizeof(double);
  for(int l1=0; l1<=L; l1++){
    double**** Tl2=table[l1];
    for(int l2=0; l2<=l1; l2++){
      double*** Tl=Tl2[l2];
      int loffset=l1-l2;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
	//cout<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl;
	double** Tm1=Tl[l-loffset];
	for(int m1=-l1; m1<=l1; m1++){
	  int m2offset=MAX(-l2,-l-m1);
	  int toread=MIN(l2,l-m1)-m2offset+1;
	  double* Tm2=Tm1[m1+l1];
	  readlength=fread(Tm2,dsize,toread,file);
	  if(readlength!=toread){cout<<" Error in reading from file "<<file<<endl; fclose(file); L=0; return -1;}
	}
      }
    }
    cout<<"."; cout.flush();
  }
  cout<<endl;

  if(fclose(file)) {cout<<" Error: Cannot close file "<<file<<endl; return -1;}

  return 1;
}


int ClebschGordan::save(char* filename){
  FILE* file=fopen(filename,"w");
  if(file==NULL) {cout<<" Error: Cannot create file "<<file<<endl; return -1;}
  int written;

  double magic=CG_MAGIC;
  written=fwrite(&magic,sizeof(double),1,file);
  if(written!=1){cout<<" Error in writing to file "<<file<<endl; fclose(file); return -1;}

  int version=CG_VERSION;
  written=fwrite(&version,sizeof(int),1,file);
  if(written!=1){cout<<" Error in writing to file "<<file<<endl; fclose(file); return -1;}

  written=fwrite(&group,sizeof(int),1,file);
  if(written!=1){cout<<" Error in writing to file "<<file<<endl; fclose(file); return -1;}

  written=fwrite(&L,sizeof(int),1,file);
  if(written!=1){cout<<" Error in writing to file "<<file<<endl; fclose(file); return -1;}

  int dsize=sizeof(double);
  for(int l1=0; l1<=L; l1++){
    double**** Tl2=table[l1];
    for(int l2=0; l2<=l1; l2++){
      double*** Tl=Tl2[l2];
      int loffset=l1-l2;
      for(int l=loffset; l<=MIN(l1+l2,L); l++){
	//cout<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl;
	double** Tm1=Tl[l-loffset];
	for(int m1=-l1; m1<=l1; m1++){
	  int m2offset=MAX(-l2,-l-m1);
	  int towrite=MIN(l2,l-m1)-m2offset+1;
	  written=fwrite(Tm1[m1+l1],dsize,towrite,file);
	  if(written!=towrite){cout<<" Error in writing to file "<<file<<endl; fclose(file); return -1;}
	}
      }
    }
  }

  if(fclose(file)) {cout<<" Error: Cannot close file "<<file<<endl; return -1;}
  return 0;
}


void ClebschGordan::print(){
  cout.precision(3);
  for(int l1=0; l1<=L; l1++){
    for(int l2=0; l2<=L; l2++){
      for(int l=abs(l1-l2); l<=MIN(l1+l2,L); l++){
        cout<<endl<<"["<<l1<<","<<l2<<","<<l<<"]"<<endl<<endl;
        for(int m1=-l1; m1<=l1; m1++){
          for(int m2=-l2; m2<=l2; m2++){
            //cout<<"* ";
            cout<<(*this)(l1,l2,l,m1,m2,m1+m2)<<" ";
          }
          cout<<endl;
        }
      }
    }
  }
}


inline int fact(int n){
  int result=1;
  for(int i=2; i<=n; i++) result*=i;
  return result;
}


inline double logfact(int n){
  double result=0;
  for(int i=2; i<=n; i++) result+=log((double)i);
  return result;
}


inline double plusminus(int k){ if(k%2==1) return -1; else return +1; }


double ClebschGordan::slowCG(int l1, int l2, int l, int m1, int m2, int m){

  int m3=-m;
  int t1=l2-m1-l;
  int t2=l1+m2-l;
  int t3=l1+l2-l;
  int t4=l1-m1;
  int t5=l2+m2;
  
  int tmin=max(0,max(t1,t2));
  int tmax=min(t3,min(t4,t5));

  double wigner=0;

  // for(int t=tmin; t<=tmax; t++)
  //   wigner += plusminus(t)/(fact(t)*fact(t-t1)*fact(t-t2)*fact(t3-t)*fact(t4-t)*fact(t5-t));
  // wigner*=plusminus(l1-l2-m3)*sqrt((double)(fact(l1+l2-l)*fact(l1-l2+l)*fact(-l1+l2+l))/fact(l1+l2+l+1));
  // wigner*=sqrt((double)(fact(l1+m1)*fact(l1-m1)*fact(l2+m2)*fact(l2-m2)*fact(l+m3)*fact(l-m3)));

  //add a log(2*l+1) to logA
  double logA=(log(2*l+1)+logfact(l+l1-l2)+logfact(l-l1+l2)+logfact(l1+l2-l)-logfact(l1+l2+l+1))/2;
  logA+=(logfact(l-m3)+logfact(l+m3)+logfact(l1-m1)+logfact(l1+m1)+logfact(l2-m2)+logfact(l2+m2))/2;

  for(int t=tmin; t<=tmax; t++){
    // cout<<t<<endl;
    //double logB=logfact(t)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t)+logfact(t-t1)+logfact(t-t2);
    double logB = logfact(t)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t)+logfact(-t1+t)+logfact(-t2+t);
    wigner += plusminus(t)*exp(logA-logB);
    }

  // cout<<"W"<<wigner<<endl;
  
  //remove sqrt(2*l+1)
  return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*wigner; 
}
/*
double ClebschGordan::slowCG(int l1, int l2, int l, int m1, int m2, int m){

  int m3=-m;
  int t1=l2-m1-l;
  int t2=l1+m2-l;
  int t3=l1+l2-l;
  int t4=l1-m1;
  int t5=l2+m2;
  
  int tmin=max(0,max(t1,t2));
  int tmax=min(t3,min(t4,t5));

  double wigner=0;



  // for(int t=tmin; t<=tmax; t++)
  //   wigner += plusminus(t)/(fact(t)*fact(t-t1)*fact(t-t2)*fact(t3-t)*fact(t4-t)*fact(t5-t));
  // wigner*=plusminus(l1-l2-m3)*sqrt((double)(fact(l1+l2-l)*fact(l1-l2+l)*fact(-l1+l2+l))/fact(l1+l2+l+1));
  // wigner*=sqrt((double)(fact(l1+m1)*fact(l1-m1)*fact(l2+m2)*fact(l2-m2)*fact(l+m3)*fact(l-m3)));

  double logA=(logfact(l1+l2-l)+logfact(l1-l2+l)+logfact(-l1+l2+l)-logfact(l1+l2+l+1))/2;
  logA+=(logfact(l1+m1)+logfact(l1-m1)+logfact(l2+m2)+logfact(l2-m2)+logfact(l+m3)+logfact(l-m3))/2;

  for(int t=tmin; t<=tmax; t++){
    // cout<<t<<endl;
    double logB=logfact(t)+logfact(t-t1)+logfact(t-t2)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t);
    wigner += plusminus(t)*exp(logA-logB);
    }

  // cout<<"W"<<wigner<<endl;
  
  return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*sqrt((double)(2*l+1))*wigner; 
}
*/