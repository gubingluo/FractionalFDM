//////////////////////////////////////////////////////////////////////////
//      2D finite difference time domain acoustic wave 
// two-order displacement wave equation forward simulation
//   multi-shots for least square reverse time migration
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "LSRTM_PP_ISO_disp2d_function.h"
#include "common.h"


/* TYPEDEFS */
typedef struct {	/* segy - trace identification header */

	int tracl;	
	int tracr;	
	int fldr;	
	int tracf;	
	int ep;		
	int cdp;	
	int cdpt;
	short trid;
	short nvs;	
	short nhs;
	short duse;
	int offset;
	int gelev;	
	int selev;	
	int sdepth;	
	int gdel;	
	int sdel;	
	int swdep;	
	int gwdep;	
	short scalel;
	short scalco;	
	int  sx;
	int  sy;	
	int  gx;	
	int  gy;	
	short counit;	
	short wevel;	
	short swevel;	
	short sut;	
	short gut;	
	short sstat;	
	short gstat;	
	short tstat;
	short laga;	
	short lagb;	
	short delrt;
	short muts;	
	short mute;	
	unsigned short ns;
	unsigned short dt;
	short gain;	
	short igc;	
	short igi;	
	short corr;	
	short sfs;
	short sfe;	
	short slen;	
	short styp;	
	short stas;	
	short stae;	
	short tatyp;	
	short afilf;	
	short afils;	
	short nofilf;	
	short nofils;	
	short lcf;	
	short hcf;	
	short lcs;	
	short hcs;	
	short year;	
	short day;	
	short hour;	
	short minute;	
	short sec;	
	short timbas;
	short trwf;	
	short grnors;	
	short grnofr;	
	short grnlof;	
	short gaps;	
	short otrav;	
#ifdef SLTSU_SEGY_H  /* begin Unocal SU segy.h differences */

	/* cwp local assignments */
	float d1;	
	float f1;	
	float d2;	
	float f2;	
	float ungpow;
	float unscale;
	short mark;	
	/* SLTSU local assignments */ 
	short mutb;
	float dz;
	float fz;
	short n2;	
     short shortpad; 
	int ntr; 	
	/* SLTSU local assignments end */ 
	short unass[8];	/* unassigned */
#else
	/* cwp local assignments */
	float d1;	
	float f1;
	float d2;
	float f2;
	float ungpow;
	float unscale;
	int ntr; 	
	short mark;	
	short shortpad; 
	short unass[14];
#endif
} suheader;



static struct 
{
	int n;  
	float c;
} nctab[NTAB] = {
{       1, 0.000052 },
{       2, 0.000061 },
{       3, 0.000030 },
{       4, 0.000053 },
{       5, 0.000066 },
{       6, 0.000067 },
{       7, 0.000071 },
{       8, 0.000062 },
{       9, 0.000079 },
{      10, 0.000080 },
{      11, 0.000052 },
{      12, 0.000069 },
{      13, 0.000103 },
{      14, 0.000123 },
{      15, 0.000050 },
{      16, 0.000086 },
{      18, 0.000108 },
{      20, 0.000101 },
{      21, 0.000098 },
{      22, 0.000135 },
{      24, 0.000090 },
{      26, 0.000165 },
{      28, 0.000084 },
{      30, 0.000132 },
{      33, 0.000158 },
{      35, 0.000138 },
{      36, 0.000147 },
{      39, 0.000207 },
{      40, 0.000156 },
{      42, 0.000158 },
{      44, 0.000176 },
{      45, 0.000171 },
{      48, 0.000185 },
{      52, 0.000227 },
{      55, 0.000242 },
{      56, 0.000194 },
{      60, 0.000215 },
{      63, 0.000233 },
{      65, 0.000288 },
{      66, 0.000271 },
{      70, 0.000248 },
{      72, 0.000247 },
{      77, 0.000285 },
{      78, 0.000395 },
{      80, 0.000285 },
{      84, 0.000209 },
{      88, 0.000332 },
{      90, 0.000321 },
{      91, 0.000372 },
{      99, 0.000400 },
{     104, 0.000391 },
{     105, 0.000358 },
{     110, 0.000440 },
{     112, 0.000367 },
{     117, 0.000494 },
{     120, 0.000413 },
{     126, 0.000424 },
{     130, 0.000549 },
{     132, 0.000480 },
{     140, 0.000450 },
{     143, 0.000637 },
{     144, 0.000497 },
{     154, 0.000590 },
{     156, 0.000626 },
{     165, 0.000654 },
{     168, 0.000536 },
{     176, 0.000656 },
{     180, 0.000611 },
{     182, 0.000730 },
{     195, 0.000839 },
{     198, 0.000786 },
{     208, 0.000835 },
{     210, 0.000751 },
{     220, 0.000826 },
{     231, 0.000926 },
{     234, 0.000991 },
{     240, 0.000852 },
{     252, 0.000820 },
{     260, 0.001053 },
{     264, 0.000987 },
{     273, 0.001152 },
{     280, 0.000952 },
{     286, 0.001299 },
{     308, 0.001155 },
{     312, 0.001270 },
{     315, 0.001156 },
{     330, 0.001397 },
{     336, 0.001173 },
{     360, 0.001259 },
{     364, 0.001471 },
{     385, 0.001569 },
{     390, 0.001767 },
{     396, 0.001552 },
{     420, 0.001516 },
{     429, 0.002015 },
{     440, 0.001748 },
{     455, 0.001988 },
{     462, 0.001921 },
{     468, 0.001956 },
{     495, 0.002106 },
{     504, 0.001769 },
{     520, 0.002196 },
{     528, 0.002127 },
{     546, 0.002454 },
{     560, 0.002099 },
{     572, 0.002632 },
{     585, 0.002665 },
{     616, 0.002397 },
{     624, 0.002711 },
{     630, 0.002496 },
{     660, 0.002812 },
{     693, 0.002949 },
{     715, 0.003571 },
{     720, 0.002783 },
{     728, 0.003060 },
{     770, 0.003392 },
{     780, 0.003553 },
{     792, 0.003198 },
{     819, 0.003726 },
{     840, 0.003234 },
{     858, 0.004354 },
{     880, 0.003800 },
{     910, 0.004304 },
{     924, 0.003975 },
{     936, 0.004123 },
{     990, 0.004517 },
{    1001, 0.005066 },
{    1008, 0.003902 },
{    1040, 0.004785 },
{    1092, 0.005017 },
{    1144, 0.005599 },
{    1155, 0.005380 },
{    1170, 0.005730 },
{    1232, 0.005323 },
{    1260, 0.005112 },
{    1287, 0.006658 },
{    1320, 0.005974 },
{    1365, 0.006781 },
{    1386, 0.006413 },
{    1430, 0.007622 },
{    1456, 0.006679 },
{    1540, 0.007032 },
{    1560, 0.007538 },
{    1584, 0.007126 },
{    1638, 0.007979 },
{    1680, 0.007225 },
{    1716, 0.008961 },
{    1820, 0.008818 },
{    1848, 0.008427 },
{    1872, 0.009004 },
{    1980, 0.009398 },
{    2002, 0.010830 },
{    2145, 0.012010 },
{    2184, 0.010586 },
{    2288, 0.012058 },
{    2310, 0.011673 },
{    2340, 0.011700 },
{    2520, 0.011062 },
{    2574, 0.014313 },
{    2640, 0.013021 },
{    2730, 0.014606 },
{    2772, 0.013216 },
{    2860, 0.015789 },
{    3003, 0.016988 },
{    3080, 0.014911 },
{    3120, 0.016393 },
{    3276, 0.016741 },
{    3432, 0.018821 },
{    3465, 0.018138 },
{    3640, 0.018892 },
{    3696, 0.018634 },
{    3960, 0.020216 },
{    4004, 0.022455 },
{    4095, 0.022523 },
{    4290, 0.026087 },
{    4368, 0.023474 },
{    4620, 0.024590 },
{    4680, 0.025641 },
{    5005, 0.030303 },
{    5040, 0.025253 },
{    5148, 0.030364 },
{    5460, 0.031250 },
{    5544, 0.029412 },
{    5720, 0.034404 },
{    6006, 0.037500 },
{    6160, 0.034091 },
{    6435, 0.040214 },
{    6552, 0.037221 },
{    6864, 0.042735 },
{    6930, 0.040214 },
{    7280, 0.042980 },
{    7920, 0.045872 },
{    8008, 0.049505 },
{    8190, 0.049834 },
{    8580, 0.055762 },
{    9009, 0.057034 },
{    9240, 0.054945 },
{    9360, 0.056818 },
{   10010, 0.066667 },
{   10296, 0.065502 },
{   10920, 0.068182 },
{   11088, 0.065217 },
{   11440, 0.075000 },
{   12012, 0.078534 },
{   12870, 0.087719 },
{   13104, 0.081081 },
{   13860, 0.084270 },
{   15015, 0.102740 },
{   16016, 0.106383 },
{   16380, 0.105634 },
{   17160, 0.119048 },
{   18018, 0.123967 },
{   18480, 0.119048 },
{   20020, 0.137615 },
{   20592, 0.140187 },
{   21840, 0.154639 },
{   24024, 0.168539 },
{   25740, 0.180723 },
{   27720, 0.180723 },
{   30030, 0.220588 },
{   32760, 0.241935 },
{   34320, 0.254237 },
{   36036, 0.254237 },
{   40040, 0.288462 },
{   45045, 0.357143 },
{   48048, 0.357143 },
{   51480, 0.384615 },
{   55440, 0.384615 },
{   60060, 0.454545 },
{   65520, 0.517241 },
{   72072, 0.576923 },
{   80080, 0.625000 },
{   90090, 0.833333 },
{  102960, 0.789474 },
{  120120, 1.153846 },
{  144144, 1.153846 },
{  180180, 1.875000 },
{  240240, 2.500000 },
{  360360, 3.750000 },
{  720720, 7.500000 },
};

int npfa (int nmin)
{
	int i;
	for (i=0; i<NTAB-1 && nctab[i].n<nmin; ++i);
	return nctab[i].n;
}
int npfar (int nmin)
{
    return 2*npfa((nmin+1)/2);
}


float sum1part(float *diffcoef, int n)
{
	float s=0;
	for (int i=0;i<n/2;i++)
	{
		s+=abs(diffcoef[i]);
	}
	return s;
}
double sum(double *data, int n)
{
	double s=0;
	for (int i=0;i<n;i++)
	{
		s=s+data[i];
	}
	return s;
}
float sum2(float **data, int nx, int nz)
{
	float s=0;
	for (int i=0;i<nz;i++)
	{
		for (int j=0;j<nx;j++)
			s=s+data[i][j];
	}
	return s;
}
void arrayabs(float *data1, float *data, int n)
{
	for (int i=0;i<n;i++)
	{
		data1[i]=fabs(data[i]);
	}
}
///////////////////---动态数组定义函数---////////////////////////////
float ***Creat3dArray(int m, int n, int k)
{
	float ***tt=(float ***)malloc(sizeof(float **)*m);
	for (int i=0;i<m;i++)
	{
		tt[i]=(float **)malloc(sizeof(float *)*n);
		for (int j=0;j<n;j++)
		{
			tt[i][j]=(float*)malloc(sizeof(float)*k);
		}
	}
	return tt;
}
void free3dArray(float ***tt, int m, int n, int k)
{
	if (tt!=NULL)
	{
		for (int i=0;i<m;i++)
		{
			for (int j=0;j<n;j++)
			{
				free((tt[i][j]));
			}
			free(tt[i]);
		}
		free(tt);
		tt=NULL;
	}
}
float **Creat2dArray(int m, int n)
{
	float **tt=(float**)malloc(sizeof(float *)*m);
	for (int i=0;i<m;i++)
	{
		tt[i]=(float*)malloc(sizeof(float)*n);
	}
	return tt;
}
void free2dArray(float **tt, int m, int n)
{
	if (tt!=NULL)
	{
		for (int i=0;i<m;i++)
		{
			free(tt[i]);
		}
		free(tt);
		tt=NULL;
	}
}
int **Creat2dArray_int(int m, int n)
{
	int **tt=(int**)malloc(sizeof(int *)*m);
	for (int i=0;i<m;i++)
	{
		tt[i]=(int*)malloc(sizeof(int)*n);
	}
	return tt;
}
void free2dArray_int(int **tt, int m, int n)
{
	if (tt!=NULL)
	{
		for (int i=0;i<m;i++)
		{
			free(tt[i]);
		}
		free(tt);
		tt=NULL;
	}
}
void fmemset1(float *p, int len)
{
	for(int j=0;j<len;j++)
		p[j]=0.0;
}

void fmemset2(float **p, int nz, int nx)
{
	for(int iz=0;iz<nz;iz++)
		for(int ix=0;ix<nx;ix++)
			p[iz][ix]=0.0;
}
void fmemset3(float ***p, int nz, int nx, int nt)
{
	for(int iz=0;iz<nz;iz++)
		for(int ix=0;ix<nx;ix++)
			for (int it=0;it<nt;it++)
				p[iz][ix][it] = 0.0;
}
void fmemcpy3(float ***dest, float ***sour, int nz, int nx, int nt)
{
	for(int iz=0;iz<nz;iz++)
		for(int ix=0;ix<nx;ix++)
			for (int it=0;it<nt;it++)
				dest[iz][ix][it] = sour[iz][ix][it];
}
void fmemset1v(float *p, int len, float v)
{
	for(int j=0;j<len;j++)
		p[j]=v;
}

void fmemset2v(float **p, int nz, int nx, float v)
{
	for(int iz=0;iz<nz;iz++)
		for(int ix=0;ix<nx;ix++)
			p[iz][ix]=v;
}
void fmemset1vp(float *p, int len, float *vp)
{
	for(int j=0;j<len;j++)
		p[j]=vp[j];
}
void fmemset2vp(float **p, int nz, int nx, float **vp)
{
	for(int iz=0;iz<nz;iz++)
		for(int ix=0;ix<nx;ix++)
			p[iz][ix]=vp[iz][ix];
}
float Maxval2(float **v, int nz, int nx)
{
	float data = v[0][0];
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			data = MAX(data,v[iz][ix]);
		}
	}
	return data;
}
float Minval2(float **v, int nz, int nx)
{
	float data = v[0][0];
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			data = MIN(data,v[iz][ix]);
		}
	}
	return data;
}
float Maxval1(float *v, int n)
{
	float a=v[0];
	for (int i=0;i<n;i++)
	{
		a=MAX(a,v[i]);
	}
	return a;
}
float Minval1(float *v, int n)
{
	float a=v[0];
	for (int i=0;i<n;i++)
	{
		a=MIN(a,v[i]);
	}
	return a;
}
float absMaxval2(float **v, int nz, int nx)
{
	float data = fabs(v[0][0]);
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			data = MAX(data,fabs(v[iz][ix]));
		}
	}
	return data;
}
float absMinval2(float **v, int nz, int nx)
{
	float data = fabs(v[0][0]);
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			data = MIN(data,fabs(v[iz][ix]));
		}
	}
	return data;
}
float absMaxval3(float ***v, int nz, int nx, int nt)
{
	float data = fabs(v[0][0][0]);
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			for (int it=0; it<nt; it++)
				data = MAX(data,fabs(v[iz][ix][it]));
		}
	}
	return data;
}
float absMinval3(float ***v, int nz, int nx, int nt)
{
	float data = fabs(v[0][0][0]);
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			for (int it=0; it<nt; it++)
				data = MIN(data,fabs(v[iz][ix][it]));
		}
	}
	return data;
}
float absMaxval2_AB(float **A, float **B, int nz, int nx)
{
	float data = fabs(A[0][0]*B[0][0]);
	for (int iz=0; iz<nz; iz++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			data = MAX(data,fabs(sqrt(A[iz][ix])*sqrt(B[iz][ix])));
		}
	}
	return data;
}
float absMaxval1(float *v, int n)
{
	float a=fabs(v[0]);
	for (int i=0;i<n;i++)
	{
		a=MAX(a,fabs(v[i]));
	}
	return a;
}
float absMinval1(float *v, int n)
{
	float a=fabs(v[0]);
	for (int i=0;i<n;i++)
	{
		a=MIN(a,fabs(v[i]));
	}
	return a;
}
void MatOperSca1(float *Matrix, float Scalar, int Operation, int n)
{
	switch(Operation)
	{
	case(1):
		for (int i=0;i<n;i++)
		{
			Matrix[i] += Scalar;
		}
		break;
	case(2):
		for (int i=0;i<n;i++)
		{
			Matrix[i] -= Scalar;
		}
		break;
	case(3):
		for (int i=0;i<n;i++)
		{
			Matrix[i] *= Scalar;
		}
		break;
	case(4):
		for (int i=0;i<n;i++)
		{
			Matrix[i] /= Scalar;
		}
		break;
	case(5):
		for (int i=0;i<n;i++)
		{
			Matrix[i] = powf(Matrix[i],Scalar);
		}
		break;
	}
}
void MatOperSca2(float **Matrix, float Scalar, int Operation, int nz, int nx)
{
	switch(Operation)
	{
	case(1):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				Matrix[iz][ix] += Scalar;
			}
		}
		break;
	case(2):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				Matrix[iz][ix] -= Scalar;
			}
		}
		break;
	case(3):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				Matrix[iz][ix] *= Scalar;
			}
		}
		break;
	case(4):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				Matrix[iz][ix] /= Scalar;
			}
		}
		break;
	case(5):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				Matrix[iz][ix] = powf(Matrix[iz][ix],Scalar);
			}
		}
		break;
	}
}
void Where1(float *Matrix, float Scalar, int Operation, int n)
{
	switch(Operation)
	{
	case(1):
		for (int i=0;i<n;i++)
		{
			if (Matrix[i] < Scalar)
				Matrix[i] = Scalar;
		}
		break;
	case(2):
		for (int i=0;i<n;i++)
		{
			if (Matrix[i] <= Scalar)
				Matrix[i] = Scalar;
		}
		break;
	case(3):
		for (int i=0;i<n;i++)
		{
			if (Matrix[i] > Scalar)
				Matrix[i] = Scalar;
		}
		break;
	case(4):
		for (int i=0;i<n;i++)
		{
			if (Matrix[i] >= Scalar)
				Matrix[i] = Scalar;
		}
		break;
	}
}
void Where2(float **Matrix, float Scalar, int Operation, int nz, int nx)
{
	switch(Operation)
	{
	case(1):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				if (Matrix[iz][ix] < Scalar)
					Matrix[iz][ix] = Scalar;
			}
		}
		break;
	case(2):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				if (Matrix[iz][ix] <= Scalar)
					Matrix[iz][ix] = Scalar;
			}
		}
		break;
	case(3):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				if (Matrix[iz][ix] > Scalar)
					Matrix[iz][ix] = Scalar;		
			}
		}
		break;
	case(4):
		for (int iz=0;iz<nz;iz++)
		{
			for (int ix=0;ix<nx;ix++)
			{
				if (Matrix[iz][ix] >= Scalar)
					Matrix[iz][ix] = Scalar;
			}
		}
		break;
	}
}
float multisum1(float *p, float signp, float *q, float signq, int len)
{
	float mysum=0.0;
	for (int i=0;i<len;i++)
		mysum += signp*p[i]*signq*q[i];
	return mysum;
}
float multisum3(float ***p, float signp, float ***q, float signq, int nz, int nx, int nshot)
{
	int iz,ix,ishot;
	float mysum=0.0;
	for(iz=0; iz<nz; iz++)
	for(ix=0; ix<nx; ix++)
	for(ishot=0; ishot<nshot; ishot++)
		mysum += signp*p[iz][ix][ishot]*signq*q[iz][ix][ishot];
	return mysum;
}
void sum2d(float **temp, int m, int n, float sum)
{
	for (int i=0;i<m;i++)
	{
		for (int j=0;j<n;j++)
		{
			sum=sum+fabs(temp[i][j]);
		}
	}
	sum=sum/m/n;
}
float sumabs(float *data, int n)
{
	float result = 0.0;
	for (int i = 0; i < n; i++)
		result += fabs(data[i]);
	return result;
}
float Maxval(float *v, int nxz)
{
	float data = v[0];
	for (int iz=0; iz<nxz; iz++)
	{

		data = MAX(data,v[iz]);
	}
	return data;
}
float Minval(float *v, int nxz)
{
	float data = v[0];
	for (int iz=0; iz<nxz; iz++)
	{

		data = MIN(data,v[iz]);
	}
	return data;
}
float absMaxval(float *v, int nxz)
{
	float data = fabs(v[0]);
	for (int iz=0; iz<nxz; iz++)
	{

		data = MAX(data,fabs(v[iz]));
	}
	return data;
}
float absMinval(float *v, int nxz)
{
	float data = fabs(v[0]);
	for (int iz=0; iz<nxz; iz++)
	{

		data = MIN(data,fabs(v[iz]));
	}
	return data;
}
void fmemset(float *input, float value, int nxz)
{
	for (int i=0;i<nxz;i++)
		input[i] = value;
}
void ReadVelMod(char velfile[40], int nx ,int nz ,float *vp)
{
	int ix,iz;
	float **vel;
	vel=Creat2dArray(nx,nz);

	FILE *fp=NULL;
	char cmd[100];
	sprintf(cmd,"./input/%s",velfile);
	fp=fopen(cmd,"rb");

	if (fp==NULL)
	{
		printf("The file %s open failed !\n",velfile);
	}
	for (ix=0;ix<nx;ix++)
	{
		fseek(fp,240L,1);
		fread(vel[ix],sizeof(float),nz,fp);
	}
	fclose(fp);

	for (iz=0;iz<nz;iz++)
		for (ix=0;ix<nx;ix++)
			vp[iz*nx+ix] = vel[ix][iz];
	free2dArray(vel,nx,nz);
}

void Diff_coeff1_displacement(float *data, int N_order)
{
	int N1=N_order/2;

	int m,i;
	for (m=1;m<=N1;m++)
	{
		float m1,i1;
		m1=(float)(m);
		float a=1.0;
		for (i=1;i<=N1;i++)
		{
			i1=(float)(i);
			a=a*i1*i1;
		}
		a=a/m1/m1;
		float b=1.0;		
		for (i=1;i<=m-1;i++)
		{
			i1=(float)(i);
			b=b*(m1*m1-i1*i1);
		}
		for (i=m+1;i<=N1;i++)
		{
			i1=(float)(i);
			b=b*(i1*i1-m1*m1);
		}
		data[m-1]=0.5*powf(-1.0,m1+1.0)*a/b/m1;
	}
}
void Diff_coeff2_displacement(float *data, int N_order)
{
	int N1=N_order/2;

	int m,i;
	for (m=1;m<=N1;m++)
	{
		float m1,i1;
		m1=(float)(m);
		float a=1.0;
		for (i=1;i<=N1;i++)
		{
			i1=(float)(i);
			a=a*i1*i1;
		}
		a=a/m1/m1;
		float b=1.0;		
		for (i=1;i<=m-1;i++)
		{
			i1=(float)(i);
			b=b*(m1*m1-i1*i1);
		}
		for (i=m+1;i<=N1;i++)
		{
			i1=(float)(i);
			b=b*(i1*i1-m1*m1);
		}
		data[m-1]=powf(-1.0,m1+1.0)*a/b/powf(m1,2.0);
	}
}
int aldle(double a[],int n,int m,double c[])
{ 
	int i,j,l,k,u,v,w,k1,k2,k3;
	double p;
	if (fabs(a[0])+1.0==1.0)
	{ 
		printf("fail\n"); 
		return(-2);
	}
	for (i=1; i<=n-1; i++)
	{ 
		u=i*n; 
		a[u]=a[u]/a[0];
	}
	for (i=1; i<=n-2; i++)
	{ 
		u=i*n+i;
		for (j=1; j<=i; j++)
		{ 
			v=i*n+j-1; 
			l=(j-1)*n+j-1;
			a[u]=a[u]-a[v]*a[v]*a[l];
		}
		p=a[u];
		if (fabs(p)+1.0==1.0)
		{ 
			printf("fail\n"); 
			return(-2);
		}
		for (k=i+1; k<=n-1; k++)
		{ 
			u=k*n+i;
			for (j=1; j<=i; j++)
			{ 
				v=k*n+j-1; 
				l=i*n+j-1; 
				w=(j-1)*n+j-1;
				a[u]=a[u]-a[v]*a[l]*a[w];
			}
			a[u]=a[u]/p;
		}
	}
	u=n*n-1;
	for (j=1; j<=n-1; j++)
	{ 
		v=(n-1)*n+j-1; 
		w=(j-1)*n+j-1;
		a[u]=a[u]-a[v]*a[v]*a[w];
	}
	p=a[u];
	if (fabs(p)+1.0==1.0)
	{ 
		printf("fail\n"); 
		return(-2);
	}
	for (j=0; j<=m-1; j++)
	for (i=1; i<=n-1; i++)
	{ 
		u=i*m+j;
		for (k=1; k<=i; k++)
		{ 
			v=i*n+k-1; 
			w=(k-1)*m+j;
			c[u]=c[u]-a[v]*c[w];
		}
	}
	for (i=1; i<=n-1; i++)
	{ 
		u=(i-1)*n+i-1;
		for (j=i; j<=n-1; j++)
		{ 
			v=(i-1)*n+j; 
			w=j*n+i-1;
			a[v]=a[u]*a[w];
		}
	}
	for (j=0; j<=m-1; j++)
	{ 
		u=(n-1)*m+j;
		c[u]=c[u]/p;
		for (k=1; k<=n-1; k++)
		{ 
			k1=n-k; 
			k3=k1-1; 
			u=k3*m+j;
			for (k2=k1; k2<=n-1; k2++)
			{ 
				v=k3*n+k2; 
				w=k2*m+j;
				c[u]=c[u]-a[v]*c[w];
			}
			c[u]=c[u]/a[k3*n+k3];
		}
	}
	return(2);
}


double fun(double beta, double alpha, int fun1_ix, int fun2_ix, int index)   
{
	double sum;
	if(index == 1)
		sum = 2.0*(1.0-cos(fun1_ix*beta))
			 *2.0*(1.0-cos(fun2_ix*beta));

	if(index == 2)
		sum = pow(beta, alpha)
			 *2.0*(1.0-cos(fun1_ix*beta));

	return(sum);
}

void fgauss(int j,double betamax,double y[])
{ 
	switch (j)
	{
		case 0: { y[0]=0.0; y[1]=betamax; break;}
		default: { }
	}
	return;
}

double fgausf(double x[], double alpha, int fun1_ix, int fun2_ix, int Index1)
{ 
	double beta;
	beta=x[0];
	return fun(beta,alpha,fun1_ix,fun2_ix,Index1);

}

double fgaus(int n,int js[], double betamax, double alpha, int fun1_ix, int fun2_ix, int Index1)
{ 
	int m,j,k,q,l,*is;
	double y[2],p,s,*x,*a,*b;
	static double t[5]={-0.9061798459,-0.5384693101,0.0,
		0.5384693101,0.9061798459};
	static double c[5]={0.2369268851,0.4786286705,0.5688888889,
		0.4786286705,0.2369268851};
	is=(int *)malloc(2*(n+1)*sizeof(int));
	x=(double *)malloc(n*sizeof(double));
	a=(double *)malloc(2*(n+1)*sizeof(double));
	b=(double *)malloc((n+1)*sizeof(double));
	m=1; l=1;
	a[n]=1.0; a[2*n+1]=1.0;
	while (l==1)
	{ 
		for (j=m;j<=n;j++)
		{
			fgauss(j-1,betamax,y);
			a[j-1]=0.5*(y[1]-y[0])/js[j-1];
			b[j-1]=a[j-1]+y[0];
			x[j-1]=a[j-1]*t[0]+b[j-1];
			a[n+j]=0.0;
			is[j-1]=1; is[n+j]=1;
		}
		j=n; q=1;
		while (q==1)
		{ 
			k=is[j-1];
			if (j==n) 
				p=fgausf(x,alpha,fun1_ix,fun2_ix,Index1);
			else 
				p=1.0;
			a[n+j]=a[n+j+1]*a[j]*p*c[k-1]+a[n+j];
			is[j-1]=is[j-1]+1;
			if (is[j-1]>5)
				if (is[n+j]>=js[j-1])
				{
					j=j-1; q=1;
					if (j==0)
					{ 
						s=a[n+1]*a[0]; free(is); free(x);
						free(a); free(b); return(s);
					}
				}
				else
				{ 
					is[n+j]=is[n+j]+1;
					b[j-1]=b[j-1]+a[j-1]*2.0;
					is[j-1]=1; k=is[j-1];
					x[j-1]=a[j-1]*t[k-1]+b[j-1];
					if (j==n) q=1;
					else q=0;
				}
				else
				{
					k=is[j-1];
					x[j-1]=a[j-1]*t[k-1]+b[j-1];
					if (j==n) q=1;
					else q=0;
				}
		}
		m=j+1;
	}
	return(1);
}

void LSequation(float *coef, int M, double betamax, double alpha)
{
	int i,k,len,row,col;
	int js[1]={200};
	double *A,*b;

	len = M;

	A=(double *)malloc(len*len*sizeof(double));
	b=(double *)malloc(len*sizeof(double));
	
	// construct B
	for (k=0; k<M; k++)
	{
		b[k] = fgaus(1,js,betamax,alpha,k+1,0,2);
	}
	// construct A upper triangular
	for (k=0; k<M; k++)
	for (i=k; i<M; i++)
	{
			A[k*len+i] = fgaus(1,js,betamax,alpha,k+1,i+1,1);
	}
	// lower triangular
	for (row=0; row<len; row++)
	for (col=0; col<row; col++)
	{
		A[row*len+col] = A[col*len+row];
	}
	// solve Ax=b x->b
	i=aldle(A, len, 1, b);
	for(i=0; i<len; i++)
 	{
 		coef[i] = (float)(b[i]);
 	}

	free(A);
	free(b);
}
void Diff_coeff2_displacement_LS(float *data, int N_order, double khmax, double alpha) // least-square solution
{
	int N1=N_order/2;

	LSequation(data, N1, khmax, alpha);
}
void Outputrecord(float *record, int nt, int nx, float dt, char buff[40], int Out_flag)
{
	int it,ix;
	FILE *fp=NULL;
	if (Out_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (it=0;it<120;it++)
		{
			header[it]=0;
		}
		header[57]=(short int)(nt);
		header[58]=(short int)(dt*1000000.0);         // dt
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);

			for (it=0; it<nt; it++)
				temp[it] = record[it*nx+ix];

			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (it=0;it<120;it++)
		{
			header[it]=0;
		}
		header[57]=(short int)(nt);
		header[58]=(short int)(dt*1000000.0);         // dt
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);
			fwrite(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	} 
}
void Inputrecord(float *record, int nt, int nx, char buff[40],int In_flag)
{
	int ix,it;
	FILE *fp=NULL;
	if (In_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[it*nx+ix] = temp[it];
		}
		fclose(fp);
		free(temp);
	} 
	else
	{
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	}	
}
void Outputimage(float **record, int nz, int nx, float dx, char buff[40], int Out_flag)
{
	int ix,iz;
	FILE *fp=NULL;
	if (Out_flag==1)
	{
		float *temp =(float *)malloc(nz*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (iz=0;iz<120;iz++)
		{
			header[iz]=0;
		}
		header[57]=(short int)(nz);
		header[58]=(short int)(dx*1000000.0);      
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);

			for (iz=0; iz<nz; iz++)
				temp[iz] = record[iz][ix];

			fwrite(temp,sizeof(float),nz,fp);
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (iz=0;iz<120;iz++)
		{
			header[iz]=0;
		}
		header[57]=(short int)(nz);
		header[58]=(short int)(dx*1000000.0);         // dt
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);
			fwrite(record[ix],sizeof(float),nz,fp);
		}
		fclose(fp);
	} 
}
void Velsmooth(float *vp, int nx, int nz, int npml)
{
	int ix,iz;
	float **vvp;
	vvp=Creat2dArray(nz,nx);
	for (ix=0;ix<npml;ix++)
	{
		for (iz=1;iz<nz-1;iz++)
		{
			vp[iz*nx + ix]=vp[iz*nx + 2*npml-ix-1];
			vp[iz*nx + nx-npml+ix]=vp[iz*nx + nx-npml-ix-1];					
		}
	}
	for (iz=0;iz<npml;iz++)
	{
		for (ix=1;ix<nx-1;ix++)
		{
			vp[iz*nx + ix]=vp[(2*npml-iz-1)*nx + ix];
			vp[(nz-npml+iz)*nx + ix]=vp[(nz-npml-iz-1)*nx + ix];			
		}
	}
	float Error=1.0;
	int Num=0;
	float MaxIter=1000;
	for (ix=0;ix<nx;ix++)
	{
		for (iz=0;iz<nz;iz++)
		{
			vvp[iz][ix]=vp[iz*nx + ix];
		}
	}
	while (Num <= MaxIter && Error >0.0005)
	{
		for (ix=1;ix<npml;ix++)
		{
			for (iz=1;iz<nz-1;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+
					vvp[iz+1][ix]+vvp[iz-1][ix]);
				
				vvp[iz][ix+nx-npml-1]=0.25*(vvp[iz][ix+nx-npml]+vvp[iz][ix+nx-npml-2]+
					vvp[iz+1][ix+nx-npml-1]+vvp[iz-1][ix+nx-npml-1]);
			}
		}
		for (ix=1;ix<nx-1;ix++)
		{
			for (iz=1;iz<npml;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+
					vvp[iz+1][ix]+vvp[iz-1][ix]);
				
				vvp[iz+nz-npml-1][ix]=0.25*(vvp[iz+nz-npml-1][ix+1]+vvp[iz+nz-npml-1][ix-1]+
					vvp[iz+nz-npml][ix]+vvp[iz+nz-npml-2][ix]);
			}
		}
		for (iz=1;iz<nz-1;iz++)
		{
			vvp[iz][0]=0.25*(vvp[iz][0]+vvp[iz][1]+vvp[iz+1][0]+vvp[iz-1][0]);
			
			vvp[iz][nx-1]=0.25*(vvp[iz][nx-1]+vvp[iz][nx-2]+vvp[iz+1][nx-1]+vvp[iz-1][nx-1]);
		}
		for (ix=1;ix<nx-1;ix++)
		{
			vvp[0][ix]=0.25*(vvp[0][ix]+vvp[1][ix]+vvp[0][ix-1]+vvp[0][ix+1]);
			
			vvp[nz-1][ix]=0.25*(vvp[nz-1][ix]+vvp[nz-2][ix]+vvp[nz-1][ix-1]+vvp[nz-1][ix+1]);
		}
		vvp[0][0]=0.25*(vvp[0][0]+vvp[1][0]+vvp[0][1]+vvp[1][1]);
		
		vvp[nz-1][0]=0.25*(vvp[nz-1][0]+vvp[nz-2][0]+vvp[nz-1][1]+vvp[nz-2][1]);
		
		vvp[0][nx-1]=0.25*(vvp[0][nx-1]+vvp[0][nx-2]+vvp[1][nx-1]+vvp[1][nx-2]);
		
		vvp[nz-1][nx-1]=0.25*(vvp[nz-1][nx-1]+vvp[nz-2][nx-1]+vvp[nz-1][nx-2]+vvp[nz-2][nx-2]);
		
		float maxvel=-1.0e+5;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				maxvel=MAX(fabs(vp[iz*nx + ix]-vvp[iz][ix])/fabs(vp[iz*nx + ix]),maxvel);
			}
		}
		Error=maxvel;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				vp[iz*nx + ix]=vvp[iz][ix];
			}
		}
		Num=Num+1;
	}
	free2dArray(vvp,nz,nx);
}
void Searchmaxgrad(int nx, int nz, int npml, float *Grad, float *maxgrad, int *nxgrad, int *nzgrad)
{
	int ix,iz;
	maxgrad[0]=-1.0e+20;
	nxgrad[0] = -1;
	nzgrad[0] = -1;
	for (iz=npml; iz<nz-npml; iz++)
	{
		for (ix=npml; ix<nx-npml; ix++)
		{
			if (fabs(Grad[(iz-npml)*(nx-2*npml)+ix-npml]) > maxgrad[0])
			{
				nxgrad[0] = ix;
				nzgrad[0] = iz;
				maxgrad[0] = fabs(Grad[(iz-npml)*(nx-2*npml)+ix-npml]);
			}
		}
	}
}
void UpdateVel(int nx, int nz, int npml, float *vpnew, float *vpold, float *Grad, float alpha)
{
	int ix, iz;
	for (iz=npml; iz<nz-npml; iz++)
	{
		for (ix=npml; ix<nx-npml; ix++)
		{
			vpnew[iz*nx+ix]=vpold[iz*nx+ix]+alpha*Grad[(iz-npml)*(nx-2*npml)+ix-npml];
		}
	}
}
void UpdateImage(int nx, int nz, int npml, float *Image, float *d, float alpha)
{
	int ix, iz;
	for (iz=npml; iz<nz-npml; iz++)
	{
		for (ix=npml; ix<nx-npml; ix++)
		{
			Image[(iz-npml)*(nx-2*npml)+(ix-npml)] += alpha*d[(iz-npml)*(nx-2*npml)+ix-npml];
		}
	}
}
void UpdateImage3d(int nz, int nx, int nshot, int startshot, int dshot, float ***Image, float ***d, double alpha)
{
	int ix, iz, ishot;
	
	for (int ishot = startshot; ishot <= nshot; ishot=ishot+dshot)
	{
		int is;
		is = (int)((ishot-startshot)/dshot);
	
		for (iz=0; iz<nz; iz++)
		for (ix=0; ix<nx; ix++)
			Image[iz][ix][is] = Image[iz][ix][is] + alpha*d[iz][ix][is];
	}
}
void Updateodcig3d(int nz, int nx, int noffset, int npml, float ***Image, float ***d, double alpha)
{
	int ix, iz, is;
	
	for (is=0; is<noffset;   is++)
	for (iz=0; iz<nz-2*npml; iz++)
	for (ix=0; ix<nx-2*npml; ix++)
		Image[iz][ix][is] = Image[iz][ix][is] + alpha*d[iz][ix][is];
}
void sincInterp(int nx, int nt0, int nt, float dt0, float dt, float *st0, float *st, float **seis0, float **seis)
{
	int j, k, m;
	float t, mysum;
	float *tt=(float *)malloc(nt0*sizeof(float)),
		*tr=(float *)malloc(nt0*sizeof(float)),
		*sinc=(float *)malloc(nt0*sizeof(float));
	for (k=0; k<nt0; k++)
	{
		tt[k] = (float)(k*dt0);
	}
	//子波插值
	tr = st0;
	for(k=0;k<nt;k++)
	{
		t = (float)(k*dt);
		mysum = 0.0;
		for (m=0; m<nt0; m++)
		{
			sinc[m] = sin(pi*(tt[m]-t)/dt0)/(pi*(tt[m]-t)/dt0);
			if (fabs(tt[m]-t) < 0.5*dt0)
				sinc[m] = 1.0;
			mysum = mysum + tr[m]*sinc[m];
		}
		st[k] = mysum;
	}
	//记录插值
	for (j=0;j<nx;j++)
	{
		tr = seis0[j];
		for(k=0;k<nt;k++)
		{
			t = (float)(k*dt);
			mysum = 0.0;
			for (m=0; m<nt0; m++)
			{
				sinc[m] = sin(pi*(tt[m]-t)/dt0)/(pi*(tt[m]-t)/dt0);
				if (fabs(tt[m]-t) < 0.5*dt0)
					sinc[m] = 1.0;
				mysum = mysum + tr[m]*sinc[m];
			}
			seis[j][k] = mysum;
		}
	}

	free(tt);
	free(tr);
	free(sinc);
}
void scaleillum_grad(float *illum, float *grad, int n, float damp)
{
	int i;
	float hmin,hmax,hcut;
	hmax = illum[0]; 
	hmin = illum[0];
	for (i=0; i<n; i++)
	{
		hmax = (hmax > illum[i])*hmax + (hmax <= illum[i])*illum[i];
		hmin = (hmin < illum[i])*hmin + (hmax >= illum[i])*illum[i];
	}
	hcut = (hmax - hmin) * damp + hmin;
	for (i=0; i<n; i++)
	{
		grad[i] = (illum[i] > hcut)*grad[i]/illum[i] + (illum[i] <= hcut)*grad[i]/hcut;
	}
}
void scaled2(float ***d, int nz, int nx, int nshot, int startshot, int dshot)
{
	int iz,ix,ishot;
	float hmax;
	
	for (int ishot = startshot; ishot <= nshot; ishot=ishot+dshot)
	{
		int is;
		is = (int)((ishot-startshot)/dshot);
				
		hmax = fabs(d[0][0][is]);
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				hmax = MAX(hmax,fabs(d[iz][ix][is]));
			}
		}
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				d[iz][ix][is] = d[iz][ix][is]/hmax;
			}
		}
	}
}
void scale_gradient(float *grad, float *illum, int nz, int nx, int npml)
{
	for (int iz=npml;iz<nz-npml;iz++)
	{
		for (int ix=npml;ix<nx-npml;ix++)
		{
			grad[(iz-npml)*(nx-2*npml)+ix-npml] /= illum[(iz-npml)*(nx-2*npml)+ix-npml];
		}
	}
}
void scale_image(float *grad, float *illum, int nz, int nx, int npml)
{

	for (int iz=npml;iz<nz-npml;iz++)
	{
		for (int ix=npml;ix<nx-npml;ix++)
		{
			grad[(iz-npml)*(nx-2*npml)+ix-npml] /= illum[(iz-npml)*(nx-2*npml)+ix-npml];
		}
	}
}
void scale_image1(float *grad, float *illumS, float *illumR, int nz, int nx, int npml)
{
	for (int iz=npml;iz<nz-npml;iz++)
	{
		for (int ix=npml;ix<nx-npml;ix++)
		{
			grad[(iz-npml)*(nx-2*npml)+ix-npml] /= sqrt(illumS[(iz-npml)*(nx-2*npml)+ix-npml])*sqrt(illumR[(iz-npml)*(nx-2*npml)+ix-npml]);
		}
	}
}
//taper the stack of all shot
void gradtaperv(int nx, int nz, int npml, int npos, float *g)
{
	int ngtsta=1;
	int ngtend=5;
	float gd=2.0;
	int nwinlen;
	int j,k;
	int ngxtstal,ngxtendl,ngxtstar,ngxtendr;
	float *wx=(float *)malloc((nx-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wx, nx-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wx[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}
	for (j=0;j<nwinlen;j++)
	{
		wx[nx-2*npml-1-j]=wx[j];
	}
	for(int iz=0;iz<nz-2*npml;iz++)
	{
		for (int ix=0;ix<nx-2*npml;ix++)
		{
			g[iz*(nx-2*npml)+ix]*=wx[ix];
		}
	}
	for(int ix=0;ix<nx-2*npml;ix++)
	{
		g[(npos-npml)*(nx-2*npml)+ix]*=0.0;
	}
	free(wx);
}
void gradtaperh1(int nx, int nz, int npml, int npos, float **g)
{
	int ngtsta=1;
	int ngtend=15;//3; Marmousi  //15 layers
	float gd=3.0;//1.0;          //3  
	int nwinlen;
	int j,k;
	float *wz=(float *)malloc((nz-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wz, nz-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wz[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}

	for(int ix=0;ix<nx-2*npml;ix++)
	{
		for (int iz=0;iz<nz-2*npml;iz++)
		{
			g[iz+npml][ix+npml] *= wz[iz];
		}
	}
	free(wz);
}
void gradtaperh(int nx, int nz, int npml, int npos, float *g)
{
	int ngtsta=1;
	int ngtend=3;//3;
	float gd=1.0;//1.0;
	int nwinlen;
	int j,k;
	float *wz=(float *)malloc((nz-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wz, nz-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wz[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}

	for(int ix=0;ix<nx-2*npml;ix++)
	{
		for (int iz=0;iz<nz-2*npml;iz++)
		{
			g[iz*(nx-2*npml)+ix]*=wz[iz];
		}
	}
	free(wz);
}
//===============================================================
void imagetaperh(int nx, int nz, int npml, int npos, float **image)
{
	int ngtsta=1;
	int ngtend=3;//3;
	float gd=1.0;//1.0;
	int nwinlen;
	int j,k;
	float *wz=(float *)malloc((nz-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wz, nz-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wz[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}

	for(int ix=0;ix<nx-2*npml;ix++)
	{
		for (int iz=0;iz<nz-2*npml;iz++)
		{
			image[ix][iz] *= wz[iz];
		}
	}
	free(wz);
}
//taper each shot
double erf(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
 
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);
 
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
 
    return sign*y;
}
void gradtaper(float **g, int nx, int nz, int npml, float dx, float dz, float spx, float spz)
{
	//taper radius for gradtaper
	int swi_shot=1;
	int filtsize=3;
	int nw=15;  //15 marmousi
	float at=1.0;
	//
	int j,k,m,n;
	int is,js;
	int ix1,ix2,jz1,jz2;
	int nwinlen;
	int npos;

	float *wgts=(float *)malloc(4*sizeof(float));
	float x,z,maxrad,rad,minw,maxw;
	float **win;
	nwinlen=MAX(nw,5);
	win=Creat2dArray(2*nwinlen+1,2*nwinlen+1);
	maxrad=sqrt(2.0)*MAX(dx,dz)*nw;

	fmemset2v(win, 2*nwinlen+1, 2*nwinlen+1, 1.0);

	for (j=0;j<2*nwinlen+1;j++)
	{
		z=(float)(j-nwinlen)*dz;
		for (k=0;k<2*nwinlen+1;k++)
		{
			x=(float)(k-nwinlen)*dx;
			rad=sqrt(x*x+z*z);
			win[j][k]=erf(at*rad/maxrad);
		}
	}

	minw = Minval2(win,2*nwinlen+1,2*nwinlen+1);
	MatOperSca2(win, minw, 2, 2*nwinlen+1, 2*nwinlen+1);
	wgts[0] = win[0][nwinlen];
	wgts[1] = win[2*nwinlen][nwinlen];
	wgts[2] = win[nwinlen][0];
	wgts[3] = win[nwinlen][2*nwinlen];
	maxw = Maxval1(wgts,4);

	MatOperSca2(win, maxw, 4, 2*nwinlen+1, 2*nwinlen+1);
	Where2(win, 1.0, 3, 2*nwinlen+1, 2*nwinlen+1);


	for (j=0;j<2*nwinlen+1;j++)
	{
		for (k=0;k<2*nwinlen+1;k++)
		{
			win[j][k]=powf(win[j][k],6.0);
		}
	}

	is=(int)(spx/dx);
	js=(int)(spz/dz);
	for (j=0;j<2*nwinlen+1;j++)
	{
		for(k=0;k<2*nwinlen+1;k++)
		{
			m=is+j-nwinlen;
			n=js+k-nwinlen;
			if((m<npml) || (m>=(nx-npml)) || (n<npml) || (n>=(nz-npml)))
				continue;
			g[n][m]=win[k][j]*g[n][m];
		}
	}
	ix1=MAX(is-filtsize,npml);
	ix2=MIN(is+filtsize,nx-npml-1);
	jz1=MAX(js-filtsize,npml);
	jz2=MIN(js+filtsize,nz-npml-1);
	for (j=ix1;j<=ix2;j++)
	{
		for (k=jz1;k<=jz2;k++)
		{
			g[k][j]=g[k][j]*0.01;
		}
	}
	free2dArray(win,2*nwinlen+1,2*nwinlen+1);
	free(wgts);
}
float Cal_epsilon(float *drecord, float n)
{
	float epsilon,delta,mu;
	int ix;
	mu = 0.0;
	for (ix=0;ix<n;ix++)
	{
		mu += drecord[ix];
	}
	mu = mu/n;
	delta = 0.0;
	for (ix=0;ix<n;ix++)
	{
		delta += (drecord[ix] - mu)*(drecord[ix] - mu);
	}
	delta = sqrtf(delta/n);
	epsilon = 0.6*delta;
	return epsilon;
}
void velsmooth(int nx, int nz, float dx, float dz, float *vel)
{
	int nvel=2;
	float lamda=20.0;
	float sigma=10.0;
	int i,j,nwxz;
	int k,i1,j1,k1,k2;
	float sigma2;

	float mysum;

	nwxz=(int)(lamda/MIN(dx,dz))+1;

	float **w,**vel1;
	w=Creat2dArray(2*nwxz+1,2*nwxz+1);
	sigma2=sigma*sigma;

	for (j=0;j<2*nwxz+1;j++)
	{
		for (k=0;k<2*nwxz+1;k++)
		{
			w[j][k]=expf(-0.5*(float)(j*j+k*k)/sigma2);
		}
	}
	MatOperSca2(w,sum2(w, 2*nwxz+1, 2*nwxz+1),4,2*nwxz+1,2*nwxz+1);

	vel1=Creat2dArray(nz,nx);
	for (int iz=0;iz<nz;iz++)
	{
		for (int ix=0;ix<nx;ix++)
		{
			vel1[iz][ix]=vel[iz*nx+ix];
		}
	}
	for (i=0;i<nx;i++)
	{
		for (j=0;j<nz;j++)
		{
			mysum = 0.0;
			for (i1=0;i1<2*nwxz+1;i1++)
			{
				for (j1=0;j1<2*nwxz+1;j1++)
				{
					k1=i+i1-nwxz;
					k2=j+j1-nwxz;
					if (k1 < 0)
						k1 = -k1-1;
					if (k1 > nx-1)
						k1 = 2*(nx-1)-k1+1;
					if (k2 < 0)
						k2 = -k2-1;
					if (k2 > nz-1)
						k2 = 2*(nz-1)-k2+1;
					mysum = mysum +w[j1][i1]*vel1[k2][k1];
				}
			}
			vel[j*nx+i] = mysum;
		}
	}
	free2dArray(w,2*nwxz+1,2*nwxz+1);
	free2dArray(vel1,nz,nx);
}
void Velsmooth(float **vp, int nx, int nz, int npml)
{
	int ix,iz;
	float **vvp;
	vvp=Creat2dArray(nz,nx);
	for (ix=0;ix<npml;ix++)
	{
		for (iz=1;iz<nz-1;iz++)
		{
			vp[iz][ix]=vp[iz][2*npml-ix-1];
			vp[iz][nx-npml+ix]=vp[iz][nx-npml-ix-1];					
		}
	}
	for (iz=0;iz<npml;iz++)
	{
		for (ix=1;ix<nx-1;ix++)
		{
			vp[iz][ix]=vp[2*npml-iz-1][ix];
			vp[nz-npml+iz][ix]=vp[nz-npml-iz-1][ix];			
		}
	}
	float Error=1.0;
	int Num=0;
	float MaxIter=1000;
	for (ix=0;ix<nx;ix++)
	{
		for (iz=0;iz<nz;iz++)
		{
			vvp[iz][ix]=vp[iz][ix];
		}
	}
	while (Num <= MaxIter && Error >0.0005)
	{
		for (ix=1;ix<npml;ix++)
		{
			for (iz=1;iz<nz-1;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+
					vvp[iz+1][ix]+vvp[iz-1][ix]);
				
				vvp[iz][ix+nx-npml-1]=0.25*(vvp[iz][ix+nx-npml]+vvp[iz][ix+nx-npml-2]+
					vvp[iz+1][ix+nx-npml-1]+vvp[iz-1][ix+nx-npml-1]);
			}
		}
		for (ix=1;ix<nx-1;ix++)
		{
			for (iz=1;iz<npml;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+
					vvp[iz+1][ix]+vvp[iz-1][ix]);
				
				vvp[iz+nz-npml-1][ix]=0.25*(vvp[iz+nz-npml-1][ix+1]+vvp[iz+nz-npml-1][ix-1]+
					vvp[iz+nz-npml][ix]+vvp[iz+nz-npml-2][ix]);
			}
		}
		for (iz=1;iz<nz-1;iz++)
		{
			vvp[iz][0]=0.25*(vvp[iz][0]+vvp[iz][1]+vvp[iz+1][0]+vvp[iz-1][0]);
			
			vvp[iz][nx-1]=0.25*(vvp[iz][nx-1]+vvp[iz][nx-2]+vvp[iz+1][nx-1]+vvp[iz-1][nx-1]);
		}
		for (ix=1;ix<nx-1;ix++)
		{
			vvp[0][ix]=0.25*(vvp[0][ix]+vvp[1][ix]+vvp[0][ix-1]+vvp[0][ix+1]);
			
			vvp[nz-1][ix]=0.25*(vvp[nz-1][ix]+vvp[nz-2][ix]+vvp[nz-1][ix-1]+vvp[nz-1][ix+1]);
		}
		vvp[0][0]=0.25*(vvp[0][0]+vvp[1][0]+vvp[0][1]+vvp[1][1]);
		
		vvp[nz-1][0]=0.25*(vvp[nz-1][0]+vvp[nz-2][0]+vvp[nz-1][1]+vvp[nz-2][1]);
		
		vvp[0][nx-1]=0.25*(vvp[0][nx-1]+vvp[0][nx-2]+vvp[1][nx-1]+vvp[1][nx-2]);
		
		vvp[nz-1][nx-1]=0.25*(vvp[nz-1][nx-1]+vvp[nz-2][nx-1]+vvp[nz-1][nx-2]+vvp[nz-2][nx-2]);
		
		float maxvel=-1.0e+5;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				maxvel=MAX(fabs(vp[iz][ix]-vvp[iz][ix])/fabs(vp[iz][ix]),maxvel);
			}
		}
		Error=maxvel;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				vp[iz][ix]=vvp[iz][ix];
			}
		}
		Num=Num+1;
	}
	free2dArray(vvp,nz,nx);
}
void vpsmooth(float *vp,int n1,int n2,int nsp)
{
	// nsp 窗口大小
	// n1 行 nz
	// n2 列 nx
	// a 速度
	// b 扩展速度
	int n1e,n2e,i1,i2,i11,i22;
	double PI=3.141592653;
	float **a,**b;
	double a1,b1,dist1,dist2;
	n1e=n1+2*nsp; //数值维度扩展
	n2e=n2+2*nsp; //数组维度扩展
	// 开辟空间  b[n1e][n2e]
	a=(float**)calloc(n1,sizeof(float*));
    for(i1=0;i1<n1;i1++)
	{
		a[i1]=(float*)calloc(n2,sizeof(float));
	}
	b=(float**)calloc(n1e,sizeof(float*));
    for(i1=0;i1<n1e;i1++)
	{
		b[i1]=(float*)calloc(n2e,sizeof(float));
	}
	for(i1=0;i1<n1;i1++)
	{
		for(i2=0;i2<n2;i2++)
		{
			a[i1][i2]=vp[i1*n2+i2];
		}
	}
	//中间
	for(i1=0;i1<n1;i1++)
	{
		for(i2=0;i2<n2;i2++)
		{
			b[i1+nsp][i2+nsp]=a[i1][i2];
		}
	}
	//左边-右边
	for(i1=0;i1<n1;i1++)
	{
		for(i2=0;i2<nsp;i2++)
		{
			b[i1+nsp][i2]=a[i1][0];
			b[i1+nsp][i2+n2+nsp]=a[i1][n2-1];
		}
	}
	//上边-下边
	for(i1=0;i1<nsp;i1++)
	{
		for(i2=0;i2<n2;i2++)
		{
			b[i1][i2+nsp]=a[0][i2];
			b[nsp+n1+i1][i2+nsp]=a[n1-1][i2];
		}
	}
	//左上角-右上角-左下角-右下角
	for(i1=0;i1<nsp;i1++)
	{
		for(i2=0;i2<nsp;i2++)
		{
			b[i1][i2]=a[0][0];
			b[i1][nsp+n2+i2]=a[0][n2-1];
			b[i1+nsp+n1][i2]=a[n1-1][0];
			b[i1+nsp+n1][i2+nsp+n2]=a[n1-1][n2-1];
		}
	}
	//
	for(i1=nsp;i1<n1+nsp;i1++)
	{
		for(i2=nsp;i2<n2+nsp;i2++)
		{
			a1=0;
			for(i11=i1-nsp;i11<=i1+nsp;i11++)
			{
				for(i22=i2-nsp;i22<=i2+nsp;i22++)
				{
					dist1=i11-i1;
					dist2=i22-i2;
					b1=exp(-(dist1*dist1+dist2*dist2)/(2.0*nsp/3.0*nsp/3.0));
					a1+=b1*b[i11][i22];
				}
			}
			a[i1-nsp][i2-nsp]=a1;
		}
	}
	a1=0;
	for(i11=0;i11<=2.0*nsp;i11++)
	{
		for(i22=0;i22<=2.0*nsp;i22++)
		{
			dist1=i11-nsp;
			dist2=i22-nsp;
			b1=exp(-(dist1*dist1+dist2*dist2)/(2.0*nsp/3.0*nsp/3.0));
			a1+=b1;
		}
	}
	for(i1=0;i1<n1;i1++)
	{
		for(i2=0;i2<n2;i2++)
		{
			a[i1][i2]/=a1;
		}
	}
	for(i1=0;i1<n1;i1++)
	{
		for(i2=0;i2<n2;i2++)
		{
			vp[i1*n2+i2] = a[i1][i2];
		}
	}
	free2dArray(a,n1,n2);
	free2dArray(b,n1e,n2e);
}
void pmlvelsmooth1d(float *vp, int nx, int nz, int npml)
{
	int ix,iz;
	float **vvp;
	vvp=Creat2dArray(nz,nx);
	for (ix=0;ix<npml;ix++)
	{
		for (iz=0;iz<nz;iz++)
		{
			vp[iz*nx+ix]=vp[iz*nx+npml];
			vp[iz*nx+nx-npml+ix]=vp[iz*nx+nx-npml-1];					
		}
	}
	for (iz=0;iz<npml;iz++)
	{
		for (ix=0;ix<nx;ix++)
		{
			vp[iz*nx+ix]=vp[npml*nx+ix];
			vp[(nz-npml+iz)*nx+ix]=vp[(nz-npml-1)*nx+ix];			
		}
	}
	float Error=1.0;
	int Num=0;
	float MaxIter=1000;
	for (ix=0;ix<nx;ix++)
	{
		for (iz=0;iz<nz;iz++)
		{
			vvp[iz][ix]=vp[iz*nx+ix];
		}
	}
	while (Num <= MaxIter && Error >0.005)
	{
		for (ix=1;ix<npml;ix++)
		{
			for (iz=1;iz<nz-1;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+vvp[iz+1][ix]+vvp[iz-1][ix]);				
				vvp[iz][ix+nx-npml-1]=0.25*(vvp[iz][ix+nx-npml]+vvp[iz][ix+nx-npml-2]+vvp[iz+1][ix+nx-npml-1]+vvp[iz-1][ix+nx-npml-1]);
			}
		}
		for (ix=1;ix<nx-1;ix++)
		{
			for (iz=1;iz<npml;iz++)
			{
				vvp[iz][ix]=0.25*(vvp[iz][ix+1]+vvp[iz][ix-1]+vvp[iz+1][ix]+vvp[iz-1][ix]);				
				vvp[iz+nz-npml-1][ix]=0.25*(vvp[iz+nz-npml-1][ix+1]+vvp[iz+nz-npml-1][ix-1]+vvp[iz+nz-npml][ix]+vvp[iz+nz-npml-2][ix]);
			}
		}
		for (iz=1;iz<nz-1;iz++)
		{
			vvp[iz][0]=0.25*(vvp[iz][0]+vvp[iz][1]+vvp[iz+1][0]+vvp[iz-1][0]);			
			vvp[iz][nx-1]=0.25*(vvp[iz][nx-1]+vvp[iz][nx-2]+vvp[iz+1][nx-1]+vvp[iz-1][nx-1]);
		}
		for (ix=1;ix<nx-1;ix++)
		{
			vvp[0][ix]=0.25*(vvp[0][ix]+vvp[1][ix]+vvp[0][ix-1]+vvp[0][ix+1]);			
			vvp[nz-1][ix]=0.25*(vvp[nz-1][ix]+vvp[nz-2][ix]+vvp[nz-1][ix-1]+vvp[nz-1][ix+1]);
		}
		vvp[0][0]=0.25*(vvp[0][0]+vvp[1][0]+vvp[0][1]+vvp[1][1]);		
		vvp[nz-1][0]=0.25*(vvp[nz-1][0]+vvp[nz-2][0]+vvp[nz-1][1]+vvp[nz-2][1]);		
		vvp[0][nx-1]=0.25*(vvp[0][nx-1]+vvp[0][nx-2]+vvp[1][nx-1]+vvp[1][nx-2]);		
		vvp[nz-1][nx-1]=0.25*(vvp[nz-1][nx-1]+vvp[nz-2][nx-1]+vvp[nz-1][nx-2]+vvp[nz-2][nx-2]);
		
		float maxvel=-1.0e+5;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				maxvel=MAX(fabs(vp[iz*nx+ix]-vvp[iz][ix])/fabs(vp[iz*nx+ix]),maxvel);
			}
		}
		Error=maxvel;
		for (iz=0;iz<nz;iz++)
		{
			for (ix=0;ix<nx;ix++)
			{
				vp[iz*nx+ix]=vvp[iz][ix];
			}
		}
		Num=Num+1;
	}
	free2dArray(vvp,nz,nx);
}
float Velmaxpml1d(float *vp, int nz, int nx, int npml)
{
	int ix,iz;
	float vml,vmr,vmt,vmd;
	float vmax;
	vml = vp[0];
	for (ix=0;ix<npml;ix++)
		for (iz=0;iz<nz;iz++)
			vml = MAX(vml,vp[iz*nx+ix]);
	vmr = vp[nx-npml];
	for (ix=nx-npml;ix<nx;ix++)
		for (iz=0;iz<nz;iz++)
			vmr = MAX(vmr,vp[iz*nx+ix]);
	vmt = vp[npml];
	for (ix=npml;ix<nx-npml;ix++)
		for (iz=0;iz<npml;iz++)
			vmt = MAX(vmt,vp[iz*nx+ix]);
	vmd = vp[(nz-npml)*nx+npml];
	for (ix=npml;ix<nx-npml;ix++)
		for (iz=nz-npml;iz<nz;iz++)
			vmd = MAX(vmd,vp[iz*nx+ix]);
	vmax = MAX(vml,vmr);
	vmax = MAX(vmax,vmt);
	vmax = MAX(vmax,vmd);

	return vmax;
}
void Order_position_2order(int *nOrderx, int *nOrderz, int npml, int nx, int nz, int Ntemp)
{
	int ix,Nx;
	int Norder = N;
	
	Nx = nx;
	
	for (ix=0;ix<Ntemp;ix++)
		nOrderx[ix] = ix;
	for (ix=Ntemp;ix<npml;ix++)
		nOrderx[ix] = Ntemp;
	for (ix=npml;ix<npml+Norder/2-Ntemp;ix++)
		nOrderx[ix] = ix-npml+Ntemp;
	for (ix=npml+Norder/2-Ntemp;ix<Nx-npml-Norder/2+Ntemp;ix++)
		nOrderx[ix] = Norder/2;
	for (ix=Nx-npml-Norder/2+Ntemp;ix<Nx-npml;ix++)
		nOrderx[ix] = Nx-npml+Ntemp-1-ix;
	for (ix=Nx-npml;ix<Nx-Ntemp-1;ix++)
		nOrderx[ix] = Ntemp;
	for (ix=Nx-Ntemp-1;ix<Nx;ix++)
		nOrderx[ix] = Nx-ix-1;
		
		
	Nx = nz;
	
	for (ix=0;ix<Ntemp;ix++)
		nOrderz[ix] = ix;
	for (ix=Ntemp;ix<npml;ix++)
		nOrderz[ix] = Ntemp;
	for (ix=npml;ix<npml+Norder/2-Ntemp;ix++)
		nOrderz[ix] = ix-npml+Ntemp;
	for (ix=npml+Norder/2-Ntemp;ix<Nx-npml-Norder/2+Ntemp;ix++)
		nOrderz[ix] = Norder/2;
	for (ix=Nx-npml-Norder/2+Ntemp;ix<Nx-npml;ix++)
		nOrderz[ix] = Nx-npml+Ntemp-1-ix;
	for (ix=Nx-npml;ix<Nx-Ntemp-1;ix++)
		nOrderz[ix] = Ntemp;
	for (ix=Nx-Ntemp-1;ix<Nx;ix++)
		nOrderz[ix] = Nx-ix-1;
}
void PML_Coeff_2order(float *ddx1, float *ddz1, float *ddx2, float *ddz2, float vpmax, int npml, int nx, int nz, float dt, float dx, float dz)
{
	int ix, iy, iz;
	float R=1.0e-3;
	int npower;
	npower = 2;

	for (ix=0;ix<npml;ix++)
	{
		ddx1[ix]   = 0.5*(npower+1)*vpmax*logf(1.0/R)*powf(1.0*(npml-ix)/npml,npower)/(npml*dx);
		ddz1[ix]   = 0.5*(npower+1)*vpmax*logf(1.0/R)*powf(1.0*(npml-ix)/npml,npower)/(npml*dz);

		ddx2[ix]  = -0.5*npower*(npower+1)*vpmax*logf(1.0/R)*powf(1.0*(npml-ix)/npml,npower-1)/(npml*npml*dx*dx);			
		ddz2[ix]  = -0.5*npower*(npower+1)*vpmax*logf(1.0/R)*powf(1.0*(npml-ix)/npml,npower-1)/(npml*npml*dz*dz);		
	}
	
	for (ix=nx-npml;ix<nx;ix++)
	{
		ddx1[ix] =  ddx1[nx-ix-1];
		ddx2[ix] = -ddx2[nx-ix-1];
	}
	for (iz=nz-npml;iz<nz;iz++)
	{
		ddz1[iz] =  ddz1[nz-iz-1];
		ddz2[iz] = -ddz2[nz-iz-1];
	}
}
void Output1d(float *record, int nt, int nx, float dt, char *buff, int Out_flag)
{
	int it,ix;
	FILE *fp=NULL;
	suheader header;
	
	header.tracl=0;header.tracr=0;header.fldr=0;header.tracf=0;header.ep=0;header.cdp=0;
	header.cdpt=0;header.trid=0;header.nvs=0;header.nhs=0;header.duse=0;header.offset=0;
	header.gelev=0;header.selev=0;header.sdepth=0;header.gdel=0;header.sdel=0;header.swdep=0;
	header.gwdep=0;header.scalel=0;header.scalco=1;header.sx=0;header.sy=0;header.gx=0;header.gy=0;
	header.counit=0;header.wevel=0;header.swevel=0;header.sut=0;header.gut=0;header.sstat=0;header.gstat=0;
	header.tstat=0;header.laga=0;header.lagb=0;header.delrt=0;header.muts=0;header.mute=0;header.ns=0;
	header.dt=0;header.gain=0;header.igc=0;header.igi=0;header.corr=0;header.sfs=0;header.sfe=0;header.slen=0;
	header.styp=0;header.stas=0;header.stae=0;header.tatyp=0;header.afilf=0;header.afils=0;header.nofilf=0;
	header.nofils=0;header.lcf=0;header.hcf=0;header.lcs=0;header.hcs=0;header.year=0;header.day=0;header.hour=0;
	header.minute=0;header.sec=0;header.timbas=0;header.trwf=0;header.grnors=0;header.grnofr=0;header.grnlof=0;
	header.gaps=0;header.otrav=0;header.d1=dt;header.f1=0;header.d2=0;header.f2=0;header.ungpow=0;header.unscale=0;
	header.mark=0;header.shortpad=0;header.ntr=0;
	
	if (Out_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		// header
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
		
		for (ix=0;ix<nx;ix++)
		{
			//header[0]=ix+1;
			//fwrite(header,2,120,fp);
			
			header.tracl=ix+1;
			header.cdp=ix+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			for (it=0; it<nt; it++)
				temp[it] = record[it*nx+ix];

			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		
		// header
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
		
		for (ix=0;ix<nx;ix++)
		{
			//header[0]=ix+1;
			//fwrite(header,2,120,fp);
						
			header.tracl=ix+1;
			header.cdp=ix+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			fwrite(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	} 
}
void Input1d(float *record, int nt, int nx, char *buff,int In_flag)
{
	int ix,it;
	FILE *fp=NULL;
	if (In_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[it*nx+ix] = temp[it];
		}
		fclose(fp);
		free(temp);
	} 
	else
	{
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	}	
}
void Output2d(float **record, int nt, int nx, float dt, char *buff, int Out_flag)
{
	int it,ix;
	FILE *fp=NULL;
	suheader header;
	
	header.tracl=0;header.tracr=0;header.fldr=0;header.tracf=0;header.ep=0;header.cdp=0;
	header.cdpt=0;header.trid=0;header.nvs=0;header.nhs=0;header.duse=0;header.offset=0;
	header.gelev=0;header.selev=0;header.sdepth=0;header.gdel=0;header.sdel=0;header.swdep=0;
	header.gwdep=0;header.scalel=0;header.scalco=1;header.sx=0;header.sy=0;header.gx=0;header.gy=0;
	header.counit=0;header.wevel=0;header.swevel=0;header.sut=0;header.gut=0;header.sstat=0;header.gstat=0;
	header.tstat=0;header.laga=0;header.lagb=0;header.delrt=0;header.muts=0;header.mute=0;header.ns=0;
	header.dt=0;header.gain=0;header.igc=0;header.igi=0;header.corr=0;header.sfs=0;header.sfe=0;header.slen=0;
	header.styp=0;header.stas=0;header.stae=0;header.tatyp=0;header.afilf=0;header.afils=0;header.nofilf=0;
	header.nofils=0;header.lcf=0;header.hcf=0;header.lcs=0;header.hcs=0;header.year=0;header.day=0;header.hour=0;
	header.minute=0;header.sec=0;header.timbas=0;header.trwf=0;header.grnors=0;header.grnofr=0;header.grnlof=0;
	header.gaps=0;header.otrav=0;header.d1=dt;header.f1=0;header.d2=0;header.f2=0;header.ungpow=0;header.unscale=0;
	header.mark=0;header.shortpad=0;header.ntr=0;
	
	if (Out_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);	
		// header
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
		
		for (ix=0;ix<nx;ix++){
			//header[0]=ix+1;
			//fwrite(header,2,120,fp);
			
			header.tracl=ix+1;
			header.cdp=ix+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			for (it=0; it<nt; it++)
				temp[it] = record[it][ix];
			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
	
		for (ix=0;ix<nx;ix++)
		{
			//header[0]=ix+1;
			//fwrite(header,2,120,fp);
			
			header.tracl=ix+1;
			header.cdp=ix+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			for (it=0; it<nt; it++)
				temp[it] = record[ix][it];
			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	} 
}
void Input2d(float **record, int nt, int nx, char *buff,int In_flag)
{
	int ix,it;
	FILE *fp=NULL;
	if (In_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[it][ix] = temp[it];
		}
		fclose(fp);
		free(temp);
	} 
	else
	{
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(&record[ix],sizeof(float),nt,fp);
		}
		fclose(fp);
	}	
}
void Output3d(float ***record, int nt, int nx, int nh, float dt, char *buff, int Out_flag)
{
	int it,ix,ih;
	FILE *fp=NULL;
	suheader header;
	
	header.tracl=0;header.tracr=0;header.fldr=0;header.tracf=0;header.ep=0;header.cdp=0;
	header.cdpt=0;header.trid=0;header.nvs=0;header.nhs=0;header.duse=0;header.offset=0;
	header.gelev=0;header.selev=0;header.sdepth=0;header.gdel=0;header.sdel=0;header.swdep=0;
	header.gwdep=0;header.scalel=0;header.scalco=1;header.sx=0;header.sy=0;header.gx=0;header.gy=0;
	header.counit=0;header.wevel=0;header.swevel=0;header.sut=0;header.gut=0;header.sstat=0;header.gstat=0;
	header.tstat=0;header.laga=0;header.lagb=0;header.delrt=0;header.muts=0;header.mute=0;header.ns=0;
	header.dt=0;header.gain=0;header.igc=0;header.igi=0;header.corr=0;header.sfs=0;header.sfe=0;header.slen=0;
	header.styp=0;header.stas=0;header.stae=0;header.tatyp=0;header.afilf=0;header.afils=0;header.nofilf=0;
	header.nofils=0;header.lcf=0;header.hcf=0;header.lcs=0;header.hcs=0;header.year=0;header.day=0;header.hour=0;
	header.minute=0;header.sec=0;header.timbas=0;header.trwf=0;header.grnors=0;header.grnofr=0;header.grnlof=0;
	header.gaps=0;header.otrav=0;header.d1=dt;header.f1=0;header.d2=0;header.f2=0;header.ungpow=0;header.unscale=0;
	header.mark=0;header.shortpad=0;header.ntr=0;
	
	if (Out_flag == 1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx*nh);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
			
		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++){
			//header[0]=ix+1;
			//header[1]=ih+1;
			//fwrite(header,2,120,fp);

			header.tracl=ix*nh+ih+1;
			header.cdp=ix+1;
			header.cdpt=ih+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			for (it=0; it<nt; it++)
				temp[it] = record[it][ix][ih];
			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
	else if (Out_flag == 2)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx*nh);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
			
		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++){
			//header[0]=ix+1;
			//header[1]=ih+1;
			//fwrite(header,2,120,fp);

			header.tracl=ix*nh+ih+1;
			header.cdp=ix+1;
			header.cdpt=ih+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			for (it=0; it<nt; it++)
				temp[it] = record[ix][it][ih];
			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	} 
	else
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		/*
		short int header[120];
		for (it=0;it<120;it++)
			header[it]=0;
		header[57]=(short int)(nt);
		if (dt < 1.0)
			header[58]=(short int)(dt*1000000.0);         // dt
		else
			header[58]=(short int)(dt*1000.0);            // dz
		header[104]=(short int)(nx*nh);
		*/
		header.ns = nt;
		if (dt < 1.0)
			header.dt=(int)(dt*1000000.0);         // dt
		else
			header.dt=(int)(dt*1000.0);            // dz
			
		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++){
			//header[0]=ix+1;
			//header[1]=ih+1;
			//fwrite(header,2,120,fp);

			header.tracl=ix*nh+ih+1;
			header.cdp=ix+1;
			header.cdpt=ih+1;
			fwrite(&header, sizeof(suheader), 1, fp);
			
			for (it=0; it<nt; it++)
				temp[it] = record[ix][ih][it];
			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
}
void Input3d(float ***record, int nt, int nx, int nh, char *buff,int In_flag)
{
	int ix,it,ih;
	FILE *fp=NULL;
	if (In_flag == 1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);

		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[it][ix][ih] = temp[it];
		}
		fclose(fp);
		free(temp);
	} 
	else if (In_flag == 2)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		
		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[ix][it][ih] = temp[it];
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
			printf("The file %s open failed !\n",buff);
		
		for (ix=0;ix<nx;ix++)
		for (ih=0;ih<nh;ih++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[ix][ih][it] = temp[it];
		}
		fclose(fp);
		free(temp);
	}	
}

void Output1D(float *record, int nt, int nx, float dt, char *buff, int Out_flag, int par_flag, int fldr, float sx0, float sz0, float gx0, float gz0, float dgx0, float offsx1, float offsx2, float DX)
{
	int it,ix;
	FILE *fp=NULL;
	suheader header;
	
	header.tracl=0;header.tracr=0;header.fldr=0;header.tracf=0;header.ep=0;header.cdp=0;
	header.cdpt=0;header.trid=0;header.nvs=0;header.nhs=0;header.duse=0;header.offset=0;
	header.gelev=0;header.selev=0;header.sdepth=0;header.gdel=0;header.sdel=0;header.swdep=0;
	header.gwdep=0;header.scalel=0;header.scalco=1;header.sx=0;header.sy=0;header.gx=0;header.gy=0;
	header.counit=0;header.wevel=0;header.swevel=0;header.sut=0;header.gut=0;header.sstat=0;header.gstat=0;
	header.tstat=0;header.laga=0;header.lagb=0;header.delrt=0;header.muts=0;header.mute=0;header.ns=0;
	header.dt=0;header.gain=0;header.igc=0;header.igi=0;header.corr=0;header.sfs=0;header.sfe=0;header.slen=0;
	header.styp=0;header.stas=0;header.stae=0;header.tatyp=0;header.afilf=0;header.afils=0;header.nofilf=0;
	header.nofils=0;header.lcf=0;header.hcf=0;header.lcs=0;header.hcs=0;header.year=0;header.day=0;header.hour=0;
	header.minute=0;header.sec=0;header.timbas=0;header.trwf=0;header.grnors=0;header.grnofr=0;header.grnlof=0;
	header.gaps=0;header.otrav=0;header.d1=dt;header.f1=0;header.d2=0;header.f2=0;header.ungpow=0;header.unscale=0;
	header.mark=0;header.shortpad=0;header.ntr=0;

	if (Out_flag == 1) // iz/it *nx +ix
	{
		if (par_flag == 1) // par_flag=1 seismic data; par_flag=0 velocity/snap
		{
			float *temp =(float *)malloc(nt*sizeof(float));
			fp=fopen(buff,"wb");
			if (fp==NULL)
				printf("The file %s open failed !\n",buff);
		
			header.ns = nt;
			if (dt < 1.0)
				header.dt=(int)(dt*1000000.0);         // dt
			else
				header.dt=(int)(dt*1000.0);            // dz
		
			for (ix=(int)(offsx1/dgx0);ix<(nx-(int)(offsx2/dgx0));ix=ix+(int)(DX/dgx0))
			{
				header.tracl=(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.fldr = fldr;
			
				header.cdp =ix+1 - (int)(offsx1/dgx0);
			
				header.selev = sz0;
				header.gelev = gz0;
			
				header.sx = sx0;
				header.gx = gx0 + ix*dgx0;
			
				fwrite(&header, sizeof(suheader), 1, fp);
			
				for (it=0; it<nt; it++)
					temp[it] = record[it*nx+ix];

				fwrite(temp,sizeof(float),nt,fp);
			}
			fclose(fp);
			free(temp);
		}
		else
		{
			float *temp =(float *)malloc(nt*sizeof(float));
			fp=fopen(buff,"wb");
			if (fp==NULL)
				printf("The file %s open failed !\n",buff);
		
			header.ns = nt;
			if (dt < 1.0)
				header.dt=(int)(dt*1000000.0);         // dt
			else
				header.dt=(int)(dt*1000.0);            // dz
		
			for (ix=(int)(offsx1/dgx0);ix<(nx-(int)(offsx2/dgx0));ix=ix+(int)(DX/dgx0))
			{
				header.tracl=(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.fldr = fldr;
			
				header.cdp =(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.selev = sz0;
				header.gelev = gz0;
			
				header.sx = gx0 + ix*dgx0;
				header.gx = gx0 + ix*dgx0;
			
				fwrite(&header, sizeof(suheader), 1, fp);
			
				for (it=0; it<nt; it++)
					temp[it] = record[it*nx+ix];

				fwrite(temp,sizeof(float),nt,fp);
			}
			fclose(fp);
			free(temp);			
		}
	}
	else    // ix*nz/nt +iz/it
	{
		if (par_flag == 1)
		{
			float *temp =(float *)malloc(nt*sizeof(float));
			fp=fopen(buff,"wb");
			if (fp==NULL)
				printf("The file %s open failed !\n",buff);
		
			header.ns = nt;
			if (dt < 1.0)
				header.dt=(int)(dt*1000000.0);         // dt
			else
				header.dt=(int)(dt*1000.0);            // dz
		
			for (ix=(int)(offsx1/dgx0);ix<(nx-(int)(offsx2/dgx0));ix=ix+(int)(DX/dgx0))
			{
				header.tracl=(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.fldr = fldr;
			
				header.cdp =ix+1 - (int)(offsx1/dgx0);
			
				header.selev = sz0;
				header.gelev = gz0;
			
				header.sx = sx0;
				header.gx = gx0 + ix*dgx0;
			
				fwrite(&header, sizeof(suheader), 1, fp);
			
				for (it=0; it<nt; it++)
					temp[it] = record[ix*nt+it];

				fwrite(temp,sizeof(float),nt,fp);
			}
			fclose(fp);
			free(temp);
		}
		else
		{
			float *temp =(float *)malloc(nt*sizeof(float));
			fp=fopen(buff,"wb");
			if (fp==NULL)
				printf("The file %s open failed !\n",buff);
		
			header.ns = nt;
			if (dt < 1.0)
				header.dt=(int)(dt*1000000.0);         // dt
			else
				header.dt=(int)(dt*1000.0);            // dz
		
			for (ix=(int)(offsx1/dgx0);ix<(nx-(int)(offsx2/dgx0));ix=ix+(int)(DX/dgx0))
			{
				header.tracl=(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.fldr = fldr;
			
				header.cdp =(ix - (int)(offsx1/dgx0))/(int)(DX/dgx0) + 1;
			
				header.selev = sz0;
				header.gelev = gz0;
			
				header.sx = gx0 + ix*dgx0;
				header.gx = gx0 + ix*dgx0;
			
				fwrite(&header, sizeof(suheader), 1, fp);
			
				for (it=0; it<nt; it++)
					temp[it] = record[ix*nt+it];

				fwrite(temp,sizeof(float),nt,fp);
			}
			fclose(fp);
			free(temp);
		}	
	}
}
void mute1x(float *seisobs, float *vp, float spx, float spz, 
		   int nt, int nx, int npml, int nw, int tlength, float dx, float dz, float dt)
{
	float vtemp,dist;
	int nsx = (int)(spx/dx) + N/2;
	int nsz = (int)(spz/dz) + N/2;
	int Nx = nx+N;
	int temp,cut;
	for (int ix=npml; ix<nx-npml; ix++)
	{
		vtemp = 0.45*(1.0/sqrtf(0.5*(vp[nsz*Nx+nsx]+vp[(nsz+1)*Nx+nsx])) + 1.0/sqrtf(0.5*(vp[(npml+N/2)*Nx+(ix+N/2)]+vp[(npml+N/2+1)*Nx+(ix+N/2)])));
		dist = (spx - ix*dx)*(spx - ix*dx) + (spz - npml*dz)*(spz - npml*dz);
	
		dist = sqrtf(dist);
		temp = tlength<nt?tlength:nt;
		cut = ((int)(dist*vtemp/dt)+nw)<nt?((int)(dist*vtemp/dt)+nw):nt;
		cut = cut>temp?cut:temp;
		cut = cut<nt?cut:nt;
		
		for (int it =0; it<cut; it++)    
			seisobs[it*(nx-2*npml)+ix-npml] = 0.0;
	}	
}
void imagetaperh3d(int nx, int nz, int npml, int nshot, int startshot, int dshot, int nwidth, float damp, float ***g)
{
	int ishot,is;
	int ngtsta=1;
	int ngtend=nwidth;
	float gd=damp;  
	int nwinlen;
	int j,k;
	float *wz=(float *)malloc((nz-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wz, nz-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wz[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}
	
	for (ishot = startshot; ishot <= nshot; ishot=ishot+dshot)
	{
		int is;
		is = (int)((ishot-startshot)/dshot);
		
		for(int ix=0;ix<nx-2*npml;ix++)
		{
			for (int iz=0;iz<nz-2*npml;iz++)
			{
				g[iz][ix][is] *= wz[iz];
			}
		}
	}
	free(wz);
}
void odcigtaperh3d(int nx, int nz, int npml, int noffset, int nwidth, float damp, float ***odcigs)
{
	int ih,ix,iz;
	int ngtsta=1;
	int ngtend=nwidth;
	float gd=damp;  
	int nwinlen;
	int j,k;
	float *wz=(float *)malloc((nz-2*npml)*sizeof(float));

	nwinlen=ngtend-ngtsta+1;
	
	fmemset1v(wz, nz-2*npml, 1.0);
	for (j=0;j<nwinlen;j++)
	{
		wz[j]=exp(-0.5*gd*gd*(j-nwinlen)*(j-nwinlen)/(4.0*nwinlen*nwinlen));
	}
	
	for (ix=0;ix<nx-2*npml;ix++)
	for (iz=0;iz<nz-2*npml;iz++)
	for (ih=0;ih<noffset;  ih++)	
		odcigs[iz][ix][ih] *= wz[iz];

	free(wz);
}
void rmNaN1d(float *Image, float value, int n)
{
	int ix;
	
	for (ix=0;ix<n;ix++)
	{
		if (!isfinite(Image[ix]))
			Image[ix] = value;
	}
}
void rmNaN2d(float **Image, float value, int n1, int n2)
{
	int i1,i2;
	
	for (i1=0;i1<n1;i1++)
	for (i2=0;i2<n2;i2++) 
	{
		if (!isfinite(Image[i1][i2]))
			Image[i1][i2] = value;
	}
}
void rmNaN3d(float ***Image, float value, int n1, int n2, int n3)
{
	int i1,i2,i3;
	
	for (i1=0;i1<n1;i1++)
	for (i2=0;i2<n2;i2++)
	for (i3=0;i3<n3;i3++) 
	{
		if (!isfinite(Image[i1][i2][i3]))
			Image[i1][i2][i3] = value;
	}
}
void slantstack(float ***adcigs, float ***odcigs, float *p, int noffset, int ntheta, int nx, int nz, float dx ,float dz)
{
	int ix,iz,ip,ih;
	int h=(noffset-1)/2;
	float b,tau,hx,t1,t3;
	int it1,it2;
	
	for (ix=h;ix<nx-h;ix++)
	{
		for (iz=0;iz<nz;iz++)
		{			
			tau = iz*dz;
			for (ip=0;ip<ntheta;ip++)
			{
				b = 0.0;
				for (ih=0;ih<noffset;ih++)
				{
					hx = (ih-h)*dx;
					t1 = (p[ip]*hx+tau)/dz;					
					it1 = (int)(floor(t1));
					t3 = t1 - it1;
					it2 = it1 + 1;
					if ((it1 < 0) || (it2 >= nz-1))
						continue;
					b += odcigs[it1][ix][ih]+(odcigs[it2][ix][ih]-odcigs[it1][ix][ih])*t3;				
				}	
				adcigs[iz][ix][ip] = b;		
			}			
		}
	}
}
void cigsmth(float ***adcigs, int nz,int nx, int np, int nsp)
{
	int npe,ip,ipl;
	int ix,iz;
	float *a,*b;
	double a1,b1,dist;
	npe=np+2*nsp;
	a=(float*)malloc(np*sizeof(float));
    	b=(float*)malloc(npe*sizeof(float));
   
	for(iz=0;iz<nz;iz++)
	for(ix=0;ix<nx;ix++)
	{		
		memset(a, 0, np*sizeof(float));
		memset(b, 0, npe*sizeof(float));
		for (ip=0;ip<np;ip++)
			b[ip+nsp] = adcigs[iz][ix][ip];
		for (ip=0;ip<nsp;ip++)
		{	b[ip] = b[nsp];
			b[ip+np+nsp] = b[np+nsp-1];
		}
		for (ip=nsp;ip<np+nsp;ip++)
		{
			a1 = 0.0;
			for (ipl=ip-nsp;ipl<=ip+nsp;ipl++)
			{
				dist = ipl - ip;
				b1 = expf(-dist*dist/(2.0*nsp*nsp));
				
				a1 += b1*b[ipl];				
			}
			a[ip-nsp] = a1;			
		}
		a1 = 0.0;
		for (ipl=0;ipl<=2*nsp;ipl++)
		{
			dist = ipl - nsp;
			b1 = expf(-dist*dist/(2.0*nsp*nsp));
			a1 += b1;					
		}
		for (ip=0;ip<np;ip++)
		{
			a[ip] /= a1;			
		}
		for (ip=0;ip<np;ip++)
			adcigs[iz][ix][ip] = a[ip];			
	}
	free(a);
	free(b);
}
void cigregu(float ***adcigs, int nz,int nx, int np, int nh, int nsp)
{
	int ix,iz,ip;
	float *b=(float *)malloc(np*(4*nsp+1)*sizeof(float));
	float *d=(float *)malloc(nz*(nx-2*nh)*np*sizeof(float));
	float **L,**A,**L1;
	float epslion=0.001;
	float *diff1 = (float *)malloc((nsp)*sizeof(float));
	float *gaus1 = (float *)malloc((2*nsp+1)*sizeof(float));
	float **diff2;	
	float **gaus2;
	
	memset(b, 0, np*(4*nsp+1)*sizeof(float));
	memset(d, 0, nz*(nx-2*nh)*np*sizeof(float));
	memset(diff1, 0, nsp*sizeof(float));
	memset(gaus1, 0, (2*nsp+1)*sizeof(float));
	
	diff2 = Creat2dArray(2*nsp+1,2*nsp+1);
	gaus2 = Creat2dArray(2*nsp+1,2*nsp+1);
	
	fmemset2(diff2, 2*nsp+1,2*nsp+1);
	fmemset2(gaus2, 2*nsp+1,2*nsp+1);
	
	Diff_coeff2_displacement(diff1, nsp*2);
	
	for (int ipl=0; ipl<nsp;ipl++){
		diff2[nsp][ipl] = diff1[nsp-ipl-1];
		diff2[nsp][nsp] -= 2.0*diff1[ipl];
		diff2[nsp][ipl+nsp+1] = diff1[ipl];
	}
	
	for (int ipl=0;ipl<nsp;ipl++)
	for (int ipr=0;ipr<nsp+ipl+1;ipr++)
		diff2[ipl][ipr] = diff2[nsp][nsp-ipl+ipr];
	for (int ipl=0;ipl<nsp;ipl++)
	for (int ipr=0;ipr<2*nsp-ipl;ipr++)
		diff2[ipl+nsp+1][ipr] =  diff2[nsp][ipr];	
	
	float a1,b1;
	a1 = 0.0;
	for (int ipl=0;ipl<2*nsp+1;ipl++)
	{
		float dist = ipl - nsp;
		b1 = expf(-dist*dist/(0.5*nsp*nsp));
		a1 += b1;	
		gaus1[ipl] = b1;				
	}
	for (int ipl=0;ipl<2*nsp+1;ipl++)
		gaus2[nsp][ipl] = gaus1[ipl]/a1;
		
	for (int ipl=0;ipl<nsp;ipl++)
	for (int ipr=0;ipr<nsp+ipl+1;ipr++)
		gaus2[ipl][ipr] = gaus2[nsp][nsp-ipl+ipr];
	for (int ipl=0;ipl<nsp;ipl++)
	for (int ipr=0;ipr<2*nsp-ipl;ipr++)
		gaus2[ipl+nsp+1][ipr] =  gaus2[nsp][ipr];	
	
	
	L = Creat2dArray(np,np);
	A = Creat2dArray(np,np);
	L1 = Creat2dArray(np,np);
	
	fmemset2(L, np, np);
	fmemset2(A, np, np);
	fmemset2(L1, np, np);
	
	for (iz=0; iz<np;iz++)
	for (ix=((iz-nsp)>0?(iz-nsp):0);ix<=(iz<(np-nsp)?(iz+nsp):(np-1));ix++)
	{
		if (iz < nsp)
		{
			L[iz][ix] = gaus2[iz][ix];
			A[iz][ix] = diff2[iz][ix];
		}
		else if (iz < np-nsp)
		{
			L[iz][ix] = gaus2[nsp][ix-iz+nsp];
			A[iz][ix] = diff2[nsp][ix-iz+nsp];
		}
		else
		{
			L[iz][ix] = gaus2[iz-np+2*nsp+1][ix-iz+nsp];
			A[iz][ix] = diff2[iz-np+2*nsp+1][ix-iz+nsp];
		}				
	}
	
	free(gaus1);
	free(diff1);
	free2dArray(gaus2,2*nsp+1,2*nsp+1);
	free2dArray(diff2,2*nsp+1,2*nsp+1);
	
	for (iz=0;iz<np;iz++)
	for (ix=0;ix<np;ix++)
	{
		a1 = 0.0;
		for (ip=0;ip<np;ip++)
			a1 += L[ip][iz]*L[ip][ix] + epslion*A[ip][iz]*A[ip][ix];	
		L1[iz][ix] = a1;
	}
	
	for (ip=0;ip<np;ip++)
	{
		if (ip < 2*nsp)
			for (int il=0;il<2*nsp+1+ip;il++)
				b[ip*(4*nsp+1)+il] = L1[ip][il];
		else if (ip < np-2*nsp)
			for (int il=0;il<4*nsp+1;il++)
				b[ip*(4*nsp+1)+il] = L1[ip][il+ip-2*nsp];
		else 
			for (int il=0;il<np+2*nsp-ip;il++)
				b[ip*(4*nsp+1)+il] = L1[ip][il+ip-2*nsp];		
	}
	
	for (iz=0;iz<nz;iz++)
	for (ix=0;ix<nx-2*nh;ix++)
	for (ip=0;ip<np;ip++)
	{
		a1=0.0;
		for (int il=0;il<np;il++)
			a1 += L[il][ip]*adcigs[iz][ix+nh][il];
				
		d[ip*nz*(nx-2*nh)+iz*(nx-2*nh)+ix] = a1;	
	}
	free2dArray(L, np, np);
	free2dArray(A, np, np);
	free2dArray(L1, np, np);
	
	int err= band(b,d,np,2*nsp,4*nsp+1,nz*(nx-2*nh));
	
	if (err > 0)
	{
		for (ip=0;ip<np;ip++)
		for (iz=0;iz<nz;iz++)
		for (ix=0;ix<nx-2*nh;ix++)		
			adcigs[iz][ix+nh][ip] = d[ip*nz*(nx-2*nh)+iz*(nx-2*nh)+ix];
	}
	free(b);
	free(d);
}
void cignorm(float ***adcigs, int nz, int nx, int np)
{
	float maxcig = absMaxval3(adcigs, nz, nx, np);
	for (int iz=0;iz<nz;iz++)
	for (int ix=0;ix<nx;ix++)
	for (int ip=0;ip<np;ip++)
		adcigs[iz][ix][ip] /= maxcig;
}

int band(float *b,float *d, int n, int l, int il, int m)
{ 	
	int ls,k,i,j,is,u,v;
    	double p,t;
    	if (il!=(2*l+1))
      	{ 
      	printf("fail\n"); 
      	return(-2);
      	}
    	ls=l;
    for (k=0;k<=n-2;k++)
      { p=0.0;
        for (i=k;i<=ls;i++)
          { t=fabs(b[i*il]);
            if (t>p) {p=t; is=i;}
          }
        if (p+1.0==1.0)
          { printf("fail\n"); return(0);}
        for (j=0;j<=m-1;j++)
          { u=k*m+j; v=is*m+j;
            t=d[u]; d[u]=d[v]; d[v]=t;
          }
        for (j=0;j<=il-1;j++)
          { u=k*il+j; v=is*il+j;
            t=b[u]; b[u]=b[v]; b[v]=t;
          }
        for (j=0;j<=m-1;j++)
          { u=k*m+j; d[u]=d[u]/b[k*il];}
        for (j=1;j<=il-1;j++)
          { u=k*il+j; b[u]=b[u]/b[k*il];}
        for (i=k+1;i<=ls;i++)
          { t=b[i*il];
            for (j=0;j<=m-1;j++)
              { u=i*m+j; v=k*m+j;
                d[u]=d[u]-t*d[v];
              }
            for (j=1;j<=il-1;j++)
              { u=i*il+j; v=k*il+j;
                b[u-1]=b[u]-t*b[v];
              }
            u=i*il+il-1; b[u]=0.0;
          }
        if (ls!=(n-1)) ls=ls+1;
      }
    p=b[(n-1)*il];
    if (fabs(p)+1.0==1.0)
      { printf("fail\n"); return(0);}
    for (j=0;j<=m-1;j++)
      { u=(n-1)*m+j; d[u]=d[u]/p;}
    ls=1;
    for (i=n-2;i>=0;i--)
      { for (k=0;k<=m-1;k++)
          { u=i*m+k;
            for (j=1;j<=ls;j++)
              { v=i*il+j; is=(i+j)*m+k;
                d[u]=d[u]-b[v]*d[is];
              }
          }
        if (ls!=(il-1)) ls=ls+1;
      }
    return(2);
  }

