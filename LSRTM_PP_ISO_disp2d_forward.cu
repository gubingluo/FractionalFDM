//////////////////////////////////////////////////////////////////////////
//      2D finite difference time domain acoustic wave 
// two-order displacement wave equation forward simulation
//   multi-shots for least square reverse time migration
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cufft.h"
#include "common.h"
#include "LSRTM_PP_ISO_disp2d_function.h"
#include "LSRTM_PP_ISO_disp2d_kernel.cuh"

int getpar(int n)
{
	int i=0,nfft,flag=1;
	while(flag)
	{
		nfft=(int)(powf((float)(2),(float)(i)));
		if (n<=nfft)
		{
			flag=0;
		}
		else
		{
			i++;
			flag=1;
		}
	};
	return nfft;
}

void extend2dx_frac(float *vel, float *vv, int nz, int nx, int Mx, int Mz, int npml, int nx1, int nx2)
{
	int iz,ix,nbr,Nz,Nx,nxpad,nzpad;

	nbr=npml+Mx/2;
	
	nxpad = nx2-nx1+1;
	nzpad = nz;
	
	Nx = nxpad+2*nbr;
	Nz = nzpad+2*nbr;

	for(ix=nx1; ix<=nx2; ix++)
	for(iz=0;   iz<nz;   iz++)
		vv[(iz+nbr)*Nx+(ix-nx1+nbr)]=vel[iz*nx+ix];
	for (iz=0;   iz<nbr;    iz++)
	for (ix=nbr; ix<Nx-nbr; ix++) 
	{
		vv[iz*Nx+ix]              = vv[ix+Nx*nbr];
		vv[(iz+nzpad+nbr)*Nx+ix]  = vv[ix+Nx*(nzpad+nbr-1)];	
	}	
	for (ix=0;   ix<nbr;   ix++)
	for (iz=0;   iz<Nz;    iz++)
	{
		vv[iz*Nx+ix]           = vv[iz*Nx+nbr];
		vv[iz*Nx+ix+nxpad+nbr] = vv[iz*Nx+nxpad+nbr-1];
	}	
}

void extract2dx(float *vel, float *vv, int nzpad, int nxpad, int Mx, int Mz, int npml)
{
	int iz,ix,nbr,Nz,Nx;

	nbr= Mx/2+npml;	
	Nx = nxpad+Mx-1;
	Nz = nzpad+Mz-1;

	for(iz=nbr; iz<Nz-nbr; iz++)
	for(ix=nbr; ix<Nx-nbr; ix++)	
		vel[(iz-nbr)*(nxpad-2*npml)+(ix-nbr)] = vv[iz*Nx+ix];	
}

void kxkz(float *kx, float *kz, int nxpad, int nzpad, float dx, float dz)
{
	int ix,iz;
	
	int Nxh = nxpad/2;
	int Nzh = nzpad/2;

	float dkx = 1.0/(nxpad*dx);
	float dkz = 1.0/(nzpad*dz);
	
	for (ix=0;ix<=Nxh;ix++)
		kx[ix] = 2.0*pi*ix*dkx;
	for (ix=Nxh+1;ix<nxpad;ix++)
		kx[ix] = (ix-nxpad)*2.0*pi*dkx;	
	
	for (iz=0;iz<=Nzh;iz++)
		kz[iz] = 2.0*pi*iz*dkz;
	for (iz=Nzh+1;iz<nzpad;iz++)
		kz[iz] = (iz-nzpad)*2.0*pi*dkz;
}

extern "C" int gpunormforward1_iso(int idevice, int nt, int nx, int nz, int sshotx, int nshotx, int dshotx, int sshotz, int nshotz, int dshotz, int NX, int npml, int imedia, int Ntemp, int ihomo,
				   float dt, float dx, float dz, float fdom, float amp, float alp, float spx0, float spz0, float dspx, float dspz, float *vp0, float *Qp0)
{
	char buffrecord[40];
	int ishot,ishotx,ishotz;
	int it,ix,iz;
	float spx,spz;
	float Spx,Spz;
	float Gpx,Gpz;
	float dx2,dz2,dt2;

	int nxpad,nzpad;
	int nsx,nsz;

	static dim3  dimBlock,dimGridp,dimGridw;
	//=============================================================================================
	// variables on host
	float *diffcoef2;
	float *vp,*Qp,*record;
	int *norderx,*norderz;
	//=============================================================================================
	// variables on device
	int *d_norderx,*d_norderz;
	float *d_wavelet,*d_diffcoef2;
	float *d_source,*d_record,*d_vp,*d_Qp;
	float *d_c1,*d_c2,*d_c3,*d_c4;
	float *d_p0,*d_p1,*d_p2,*d_pt;
	float *d_w1,*d_w2, *d_w3;
	float *d_wt1,*d_wt2, *d_wt3;	
	cufftComplex *d_wk1,*d_wk2,*d_wk3;
	//=============================================================================================
	//参数换算
	dx2 = dx*dz;
	dz2 = dz*dz;
	dt2 = dt*dt;

	diffcoef2=(float *)malloc((N/2)*(N/2)*sizeof(float));
	memset(diffcoef2,0,(N/2)*(N/2)*sizeof(float));
	for (ix=0;ix<N/2;ix++)
	{
		int N1 = 2*(ix+1);
		float *diff2temp = (float *)malloc((N1/2)*sizeof(float));
//		Diff_coeff2_displacement(diff2temp, N1);
		Diff_coeff2_displacement_LS(diff2temp, N1, 2.8, 2.0);

		for (iz=0;iz<N1/2;iz++)
		{
			diffcoef2[ix*N/2 + iz] = diff2temp[iz];
		}
		free(diff2temp);
	}
	//=============================================================================================
	cudaSetDevice(idevice);
	check_gpu_error("Failed to initialize device");	
	cudaMalloc(&d_wavelet,       nt*sizeof(float));
	cudaMemset(d_wavelet,   0,   nt*sizeof(float));
	cuda_ricker_wavelet<<<(nt+511)/512,512>>>(d_wavelet, fdom, dt, nt);
	
	for (ishotx = sshotx; ishotx <= nshotx; ishotx=ishotx+dshotx)
	for (ishotz = sshotz; ishotz <= nshotz; ishotz=ishotz+dshotz)
	{	
		ishot = (ishotx-1)*nshotz+ishotz;
		//=============================================================================================
		//局部小排列
		int is,np,nr,sp1,nx1,nx2;
		int Nx,Nz;
		int Mx,Mz;
		float phi;
		float alpha1,alpha2,alpha3;

		sp1 = (int)(spx0/dx);
		np = (int)(dspx/dx);
		nr = (NX-1)/2;
		is = sp1 + (ishotx-1)*np;
			
		nx1 = MAX(0,is-nr);
		nx2 = MIN(nx-1,is+nr);

		spx = (is-nx1)*dx + npml*dx;
		spz = spz0 + (ishotz-1)*dspz + npml*dz;
		
		Spx = is*dx;
		Spz = spz0 + (ishotz-1)*dspz ;
		
		Gpx = nx1*dx;
		Gpz = 0.0;

		nxpad = nx2 - nx1 + 1 + 2*npml;
		nzpad = nz + 2*npml;
		
		Mx = Mxz; // N + NN + 1;
		Mz = Mxz; // N + NN + 1;
		
		Nx = nxpad + Mx-1;
		Nz = nzpad + Mz-1;

		nsx = (int)(spx/dx);				
		nsz = (int)(spz/dz);
		//=============================================================================================
		dimBlock = dim3(Block_Sizez, Block_Sizex);
		dimGridp = dim3((nzpad+Block_Sizez-1)/Block_Sizez, (nxpad+Block_Sizex-1)/Block_Sizex);
		dimGridw = dim3((Mz   +Block_Sizez-1)/Block_Sizez, (Mx   +Block_Sizex-1)/Block_Sizex);
		//=============================================================================================
		//模拟参数换算与准备
		record = (float *)malloc(nt*(nxpad-2*npml)*sizeof(float));
		vp     = (float *)malloc(Nz*Nx*sizeof(float));
		Qp     = (float *)malloc(Nz*Nx*sizeof(float));
		norderx  = (int *)malloc(nxpad*sizeof(int));
		norderz  = (int *)malloc(nzpad*sizeof(int));
		
		memset(record,  0, nt*(nxpad-2*npml)*sizeof(float));
		memset(vp,      0, Nz*Nx*sizeof(float));
		memset(Qp,      0, Nz*Nx*sizeof(float));
		memset(norderx, 0, nxpad*sizeof(int));
		memset(norderz, 0, nzpad*sizeof(int));
		//=============================================================================================
		extend2dx_frac(vp0, vp, nz, nx, Mx, Mz, npml, nx1, nx2);
		extend2dx_frac(Qp0, Qp, nz, nx, Mx, Mz, npml, nx1, nx2);
			
		phi = 1.003/(pi*Qp[0]);
		for (iz=0;iz<Nz;iz++)
		for (ix=0;ix<Nx;ix++){
			vp[iz*Nx+ix] = vp[iz*Nx+ix]*vp[iz*Nx+ix];			
			Qp[iz*Nx+ix] = 1.0/(pi*Qp[iz*Nx+ix]);
			
			phi = MAX(phi,1.003*Qp[iz*Nx+ix]);
		}
		// homogeneous
		if (ihomo == 1)
		{
			float qp,vv;
			printf("please input v and q:\n");
			scanf("%f%f",&vv,&qp);

			phi = 1.003/(pi*qp);
			for (iz=0;iz<Nz;iz++)
			for (ix=0;ix<Nx;ix++){
				vp[iz*Nx+ix] = vv*vv;
				Qp[iz*Nx+ix] = 1.0/(pi*qp);
			}
		}
						
		alpha1 = 2.0;
		alpha2 = 2.0*phi+2.0;
		alpha3 = 1.0;
		//=============================================================================================		
		cudaMalloc(&d_p0,			Nx*Nz*sizeof(float));
		cudaMalloc(&d_p1,			Nx*Nz*sizeof(float));
		cudaMalloc(&d_p2,			Nx*Nz*sizeof(float));	
		cudaMalloc(&d_pt,			Nx*Nz*sizeof(float));		
		cudaMemset(d_p0,     0,		Nx*Nz*sizeof(float));
		cudaMemset(d_p1,     0,		Nx*Nz*sizeof(float));
		cudaMemset(d_p2,     0,		Nx*Nz*sizeof(float));
		cudaMemset(d_pt,     0,		Nx*Nz*sizeof(float));
		//=============================================================================================		
		cudaMalloc(&d_wk1,			Mx*Mz*sizeof(cufftComplex));
		cudaMalloc(&d_wk2,			Mx*Mz*sizeof(cufftComplex));
		cudaMalloc(&d_wk3,			Mx*Mz*sizeof(cufftComplex));
		cudaMemset(d_wk1,    0,		Mx*Mz*sizeof(cufftComplex));
		cudaMemset(d_wk2,    0,		Mx*Mz*sizeof(cufftComplex));
		cudaMemset(d_wk3,    0,		Mx*Mz*sizeof(cufftComplex));

		cudaMalloc(&d_w1,			N/2*Mx*Mz*sizeof(float));
		cudaMalloc(&d_w2,			N/2*Mx*Mz*sizeof(float));
		cudaMalloc(&d_w3,			N/2*Mx*Mz*sizeof(float));	
		cudaMemset(d_w1,     0,		N/2*Mx*Mz*sizeof(float));
		cudaMemset(d_w2,     0,		N/2*Mx*Mz*sizeof(float));
		cudaMemset(d_w3,     0,		N/2*Mx*Mz*sizeof(float));
		
		cudaMalloc(&d_wt1,			N/2*Mx*Mz*sizeof(float));
		cudaMalloc(&d_wt2,			N/2*Mx*Mz*sizeof(float));
		cudaMalloc(&d_wt3,			N/2*Mx*Mz*sizeof(float));	
		cudaMemset(d_wt1,     0,	N/2*Mx*Mz*sizeof(float));
		cudaMemset(d_wt2,     0,	N/2*Mx*Mz*sizeof(float));
		cudaMemset(d_wt3,     0,	N/2*Mx*Mz*sizeof(float));
		//=============================================================================================	
		cudaMalloc(&d_record,     	nt*(nxpad-2*npml)*sizeof(float));
		cudaMemset(d_record,    0,  	nt*(nxpad-2*npml)*sizeof(float));
		//=============================================================================================	
		cudaMalloc(&d_diffcoef2,  (N/2)*(N/2)*sizeof(float));		
		cudaMalloc(&d_vp,         Nx*Nz*sizeof(float));
		cudaMalloc(&d_Qp,         Nx*Nz*sizeof(float));
		cudaMemcpy(d_diffcoef2, diffcoef2, 	(N/2)*(N/2)*sizeof(float), 	cudaMemcpyHostToDevice);			
		cudaMemcpy(d_vp,   	vp,    		Nx*Nz*sizeof(float), 		cudaMemcpyHostToDevice);
		cudaMemcpy(d_Qp,   	Qp,    		Nx*Nz*sizeof(float), 		cudaMemcpyHostToDevice);
		//=============================================================================================			
		cudaMalloc(&d_c1,			nxpad*nzpad*sizeof(float));
		cudaMalloc(&d_c2,			nxpad*nzpad*sizeof(float));
		cudaMalloc(&d_c3,			nxpad*nzpad*sizeof(float));
		cudaMalloc(&d_c4,			nxpad*nzpad*sizeof(float));
		cudaMemset(d_c1,      0,	nxpad*nzpad*sizeof(float));
		cudaMemset(d_c2,      0,	nxpad*nzpad*sizeof(float));
		cudaMemset(d_c3,      0,	nxpad*nzpad*sizeof(float));
		cudaMemset(d_c4,      0,	nxpad*nzpad*sizeof(float));
		cuda_C1ToC4<<<dimGridp,dimBlock>>>(d_c1, d_c2, d_c3, d_c4, d_Qp, phi, nxpad, nzpad);
		//=============================================================================================
		Order_position_2order(norderx, norderz, npml, nxpad, nzpad, N/2);//Ntemp);			
		cudaMalloc(&d_norderx,     nxpad*sizeof(int));
		cudaMalloc(&d_norderz,     nzpad*sizeof(int));
		cudaMemcpy(d_norderx,  	norderx,  	nxpad*sizeof(int),     		cudaMemcpyHostToDevice);
		cudaMemcpy(d_norderz,  	norderz,  	nzpad*sizeof(int),     		cudaMemcpyHostToDevice);
		free(norderx);free(norderz);
		//=============================================================================================
		cudaMalloc(&d_source,     		Nx*Nz*sizeof(float));
		cudaMemset(d_source,    0,  	Nz*Nx*sizeof(float));	
		cuda_source<<<dimGridp,dimBlock>>>(d_source, nsx, nsz, nxpad, nzpad, Mx, Mz, amp, alp, dx2, dz2);
		//=============================================================================================
		int cflag = 0;
		if (cflag == 0)
		{
			cufftHandle plan_forward,plan_backward;
			cufftPlan2d(&plan_forward, Mz,Mx,CUFFT_C2C);
			cufftPlan2d(&plan_backward,Mz,Mx,CUFFT_C2C);				
			for (int in=1;in<=N/2;in++)
			{
				cuda_FracFDCoef<<<dimGridw,dimBlock>>>(d_wk1, d_wk2, d_wk3, d_diffcoef2, alpha1, alpha2, alpha3, Mx, Mz, in);
				cufftExecC2C(plan_backward, d_wk1, d_wk1, CUFFT_INVERSE);
				cufftExecC2C(plan_backward, d_wk2, d_wk2, CUFFT_INVERSE);
				cufftExecC2C(plan_backward, d_wk3, d_wk3, CUFFT_INVERSE);
				cudacpyw<<<dimGridw,dimBlock>>>(d_wk1, d_wk2, d_wk3, d_wt1, d_wt2, d_wt3, &d_w1[(in-1)*Mx*Mz], &d_w2[(in-1)*Mx*Mz], &d_w3[(in-1)*Mx*Mz], Mz, Mx, in);
			}		
			cufftDestroy(plan_forward);
			cufftDestroy(plan_backward);
		}
		else
		{
			for (int in=1;in<=N/2;in++)
			{
				float *w1,*w2,*w3,*W1,*W2,*W3;
				w1 = (float *)malloc((2*in+1)*(2*in+1)*sizeof(float));
				w2 = (float *)malloc((2*in+1)*(2*in+1)*sizeof(float));
				w3 = (float *)malloc((2*in+1)*(2*in+1)*sizeof(float));
				W1 = (float *)malloc(Mx*Mz*sizeof(float));
				W2 = (float *)malloc(Mx*Mz*sizeof(float));
				W3 = (float *)malloc(Mx*Mz*sizeof(float));

				memset(W1,      0, Mx*Mz*sizeof(float));
				memset(W2,      0, Mx*Mz*sizeof(float));
				memset(W3,      0, Mx*Mz*sizeof(float));

				sprintf(buffrecord,"./coef/Coef_M%d_alpha1.su",in);
				Input1d(w1, 2*in+1, 2*in+1, buffrecord, 1);
				sprintf(buffrecord,"./coef/Coef_M%d_alpha2.su",in);
				Input1d(w2, 2*in+1, 2*in+1, buffrecord, 1);
				sprintf(buffrecord,"./coef/Coef_M%d_alpha3.su",in);
				Input1d(w3, 2*in+1, 2*in+1, buffrecord, 1);

				for (ix=Mx/2-in; ix<=Mx/2+in; ix++)
				{
					for (iz=Mz/2-in; iz<=Mz/2+in; iz++)
					{
						W1[iz*Mx+ix] = w1[(iz-Mz/2+in)*(2*in+1)+(ix-Mx/2+in)];
						W2[iz*Mx+ix] = w2[(iz-Mz/2+in)*(2*in+1)+(ix-Mx/2+in)];
						W3[iz*Mx+ix] = w3[(iz-Mz/2+in)*(2*in+1)+(ix-Mx/2+in)];
					}
				}
				cudaMemcpy(&d_w1[(in-1)*Mx*Mz],   W1,  Mx*Mz*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(&d_w2[(in-1)*Mx*Mz],   W2,  Mx*Mz*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(&d_w3[(in-1)*Mx*Mz],   W3,  Mx*Mz*sizeof(float), cudaMemcpyHostToDevice);

				free(w1);
				free(w2);
				free(w3);
				free(W1);
				free(W2);
				free(W3);
			}			
		}	
		cudaFree(d_wk1);cudaFree(d_wk2);cudaFree(d_wk3);
		cudaFree(d_wt1);cudaFree(d_wt2);cudaFree(d_wt3);			
		//=============================================================================================
		float *tt= (float *)malloc(Nx*Nz*sizeof(float));
		float *p=(float *)malloc((nxpad-2*npml)*(nzpad-2*npml)*sizeof(float));
		for (it=0; it<nt; it++)
		{
			cuda_record<<<(nxpad-2*npml+127)/128,128>>>(d_p1, &d_record[it*(nxpad-2*npml)], npml, nxpad, Mx, Mz);			
			cuda_forward_p_iso<<<dimGridp,dimBlock>>>(d_p2, d_p1, d_p0, d_pt, d_w1, d_w2, d_w3, d_vp, d_Qp, d_norderx, d_norderz, 
								alpha1, alpha2, alpha3, d_c1, d_c2, d_c3, d_c4, dt, dx, dz, fdom, npml, nxpad, nzpad, Mx, Mz);
			cuda_abc<<<dimGridp,dimBlock>>>(d_p2, d_p1, d_p0, d_vp, nxpad, nzpad, Mx, Mz, npml, dx, dz, dt);
			cuda_add_source<<<dimGridp,dimBlock>>>(d_p2, d_source, d_vp, d_wavelet, dt2, 1, nxpad, nzpad, Mx, Mz, it);
			
			cudaMemcpy(d_p0,   d_p1,    Nx*Nz*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_p1,   d_p2,    Nx*Nz*sizeof(float), cudaMemcpyDeviceToDevice);
			
			if (it%500 == 0 && it != 0)
			{
				cudaMemcpy(tt,  d_p1,   Nx*Nz*sizeof(float), cudaMemcpyDeviceToHost);	
				sprintf(buffrecord,"./seisReal/%dseis-snap-ISO-NCQ-FD-Forder.su",it);
				extract2dx(p, tt, nzpad, nxpad, Mx, Mz, npml);
				
				Output1D(p, nzpad-2*npml, nxpad-2*npml, dz, buffrecord, 1, 0, it, Spx, Spz, Gpx, Gpz, dx, 0.0, 0.0, dx);	
			}				
		}
		free(p);
		free(tt);		
		//=============================================================================================
		cudaMemcpy(record,  d_record,    nt*(nxpad-2*npml)*sizeof(float), cudaMemcpyDeviceToHost);
		sprintf(buffrecord,"./seisReal/%dseisreal.su",ishot);
		Output1D(record, nt, nxpad-2*npml, dt, buffrecord, 1, 1, ishot, Spx, Spz, Gpx, Gpz, dx, 0.0, 0.0, dx);	
		//=============================================================================================
		free(vp);free(Qp);free(record);		
		//=============================================================================================
		cudaFree(d_record);	cudaFree(d_source);	cudaFree(d_diffcoef2);	
		cudaFree(d_vp);		cudaFree(d_Qp);
		cudaFree(d_c1);		cudaFree(d_c1);		cudaFree(d_c3);		cudaFree(d_c4);
			
		cudaFree(d_p0);		cudaFree(d_p1);		cudaFree(d_p2);		cudaFree(d_pt);			
		cudaFree(d_w1);		cudaFree(d_w2);		cudaFree(d_w3);	
		cudaFree(d_norderx);	cudaFree(d_norderz);	
	}
	cudaFree(d_wavelet);
	free(diffcoef2);

	return 0;
}
