#include<fstream>
#include<cmath>
#include"rk4.h"
using namespace std;

double Xa=0.708, Za=0.02;
double h=1e-3, mu_a=0.686, xi1=8.1611, n=3.;
double eps=1e-3;

double X(double x)
{
	double a2=4.,
		   a4=15.;
		   
	return Xa*(a2*x*x+a4*x*x*x*x)/(1.+a2*x*x+a4*x*x*x*x);	   
}

double Y(double x)
{
	return 1.-X(x)-Za;
}

double mu(double x)
{
	return 1./(2.*X(x)+0.75*Y(x)+0.5*Za);
}

double f(double x)
{
	return mu(x)/mu_a;
}

double ff(double x, double eps)
{
	return (f(x+eps)-f(x-eps))/2./eps;
}

double Phi(double x)
{
	double  a1=-0.00228434,
			a2=1.44883,
			a3=-0.21554,
			a4=0.0188013,
			b1=4.16376,
			b2=0.483279,
			b3=2.71489,
			b4=2.21625;

	return (a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x)/(b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x);
}

double df(int i, double x, double *y)
{
	switch(i)
	{
		case 0: return y[1];
		case 1: return -2./x*y[1]-pow(y[0],n)*f(x/xi1)*f(x/xi1)-f(0.)/xi1*ff(x/xi1,eps)*Phi(x);
		
		
	}
	return 0;
}

int main()
{
	
	ofstream out ("new_res_m=0.68.txt");
	out.precision(5);
	out<<"#mu(0)="<<mu(0.)/mu_a<<endl;
	double y[2], xi, yold, pyold;
	init_rk(2);
	
	y[0]=1;
	y[1]=0;
		
	for(xi=1e-3; y[0]>0; xi=rk4(xi,df,y,h))
	{
		out<<xi<<'\t'<<y[0]<<'\t'<<y[1]<<endl;
		yold=y[0];
		pyold=y[1];
		
	}
	xi-=h+yold/pyold;
	y[0]=yold;
	y[1]=pyold;

	close_rk();

	return 0;
}