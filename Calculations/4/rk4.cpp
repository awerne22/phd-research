#include "rk4.h"

double *ny,*nny,*k1,*k2,*k3;
int rk_N = 0;

void init_rk(int n)
{
	rk_N = n;
	ny = new double [n];
	nny = new double [n];
	k1 = new double [n];
	k2 = new double [n];
	k3 = new double [n];
}

void close_rk()
{
	rk_N = 0;
	delete [] ny, nny, k1, k2, k3;
}

double rk4(double x, double (*f)(int, double, double *),double *vy, double h)
{
	int i;
	
	for (i = 0; i < rk_N; i++)
	{
		k1[i] = h*f(i, x, vy);
		ny[i] = vy[i] + k1[i]/2.0;
	}
	
	for (i = 0; i < rk_N; i++)
	{
		k2[i] = h*f(i, x+h/2.0, ny);
		nny[i] = vy[i] + k2[i]/2.0;
	}
	
	for (i = 0; i < rk_N; i++){
		k3[i] = h*f(i, x+h/2.0, nny);
		ny[i] = vy[i] + k3[i];
	}
	
	for (i = 0; i < rk_N; i++)
		vy[i] += (k1[i] + (k2[i] + k3[i])*2.0 + h*f(i, x+h, ny))/6.0;
		
	return x + h;
}
