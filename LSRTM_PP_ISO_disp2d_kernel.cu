//////////////////////////////////////////////////////////////////////////
//      2D finite difference time domain acoustic wave 
// two-order displacement wave equation forward simulation
//   multi-shots for least square reverse time migration
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cufft.h"
#include "common.h"


//=========================================================================================================
__global__ void cuda_ricker_wavelet(float *d_wavelet, float fdom, float dt, int nt)
{
	int it = threadIdx.x + blockDim.x*blockIdx.x;
	float temp = pi*fdom*fabs(it*dt - 1.0/fdom);
	temp *=temp;
	if (it < nt){
          d_wavelet[it] = (1.0 - 2.0*temp)*expf(-temp);}
}
//=========================================================================================================
__global__ void cuda_source(float *d_source, int nsx, int nsz, int nxpad, int nzpad, int Mx, int Mz, float amp, float alp, float dx2, float dz2)
{
	int idz = threadIdx.x + blockDim.x*blockIdx.x + Mz/2; //(radius+NN/2);
	int idx = threadIdx.y + blockDim.y*blockIdx.y + Mx/2; //(radius+NN/2);
	
	int Nx = nxpad+Mx-1;
	float x,z;
	float dist;

	if (idx < nxpad+Mx/2 && idz < nzpad+Mz/2)
	{
		x = (float)(idx - Mx/2 - nsx);
		z = (float)(idz - Mz/2 - nsz);
		dist = x*x*dx2+z*z*dz2;
		
          	d_source[idz*Nx+idx] = amp*expf(-alp*alp*dist);
        }
}
//=========================================================================================================
__global__ void cuda_add_source(float *d_p2, float *d_source, float *d_vp, float *d_wlt, float dt2, int add, int nxpad, int nzpad, int Mx, int Mz, int it)
{
	int idz = threadIdx.x + blockDim.x*blockIdx.x + Mz/2; //(radius+NN/2);
	int idx = threadIdx.y + blockDim.y*blockIdx.y + Mx/2; //(radius+NN/2);
	
	int Nx = nxpad+Mx-1;
	
	int id = idz*Nx+idx;
	
	if (idx < nxpad+Mx/2 && idz < nzpad+Mz/2)
		d_p2[id] += add*dt2*d_vp[id]*d_source[id]*d_wlt[it];
}
//=========================================================================================================
__global__ void cuda_record(float *d_p, float *d_seis, int npml, int nxpad, int Mx, int Mz)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int Nx = nxpad+Mx-1;
	int igx = npml+Mx/2+idx;
	int igz = npml+Mz/2;
		
	if (idx < nxpad - 2*npml)      
		d_seis[idx] = d_p[igz*Nx + igx];
}
__global__ void cuda_insert_record(float *d_p, float *d_vp, float *d_seis, int npml, int nxpad, int Mx, int Mz, float dt2)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int Nx = nxpad+Mx-1;
	int igx = npml+Mx/2+idx;
	int igz = npml+Mz/2;	
	
	if (idx < nxpad - 2*npml)      
		d_p[igz*Nx + igx] += dt2*d_vp[igz*Nx + igx]*d_seis[idx];
}
//==============================================================================
__global__ void cuda_C1ToC4(float *d_c1, float *d_c2, float *d_c3, float *d_c4, float *d_gm, float m, int nxpad, int nzpad)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;	
	int Nx = nxpad+N;
	int id1,id2;  	// x-space
	int ik,nk = 501;	
	float k,dk,gm;
	dk = pi/nk;

	float A11,A12,A13,B11,B12,B13;
	float A21,A22,A23,B21,B22,B23;

	if (idz < nzpad && idx < nxpad)
	{
		id1 = idz*nxpad+idx; 		// k-space
		id2 = (idz+radius)*Nx+(idx+radius);  	// x-space

		gm = d_gm[id2];

		A11 = 0.0; A12 = 0.0; A13 = 0.0;
		B11 = 0.0; B12 = 0.0; B13 = 0.0;		
		A21 = 0.0; A22 = 0.0; A23 = 0.0;
		B21 = 0.0; B22 = 0.0; B23 = 0.0;

		for (ik=0; ik<nk; ik++)
		{
			k = ik*dk;
				
			A11 += gm*gm*k*k;
			A12 += gm*gm*k*k*k;
			A13 += gm*k*k-gm*powf(k,2.0+2.0*gm);	
			B11 += gm*gm*k*k*k;
			B12 += gm*gm*k*k*k*k;
			B13 += gm*k*k*k-gm*powf(k,3.0+2.0*gm);
				
			A21 += gm*gm*k*k*k*k;
			A22 += gm*gm*powf(k,4.0+2.0*m);
			A23 += gm*k*k*k*k-gm*powf(k,4.0+2.0*gm);
			B21 += gm*gm*powf(k,4.0+2.0*m);
			B22 += gm*gm*powf(k,4.0+4.0*m);
			B23 += gm*powf(k,4.0+2.0*m)-gm*powf(k,4.0+2.0*gm+2.0*m);
		}
		d_c1[id1] = (A12*B13-A13*B12)/(A11*B12-A12*B11);
		d_c2[id1] = (A11*B13-A13*B11)/(A12*B11-A11*B12);	
		d_c3[id1] = (A22*B23-A23*B22)/(A21*B22-A22*B21);
		d_c4[id1] = (A21*B23-A23*B21)/(A22*B21-A21*B22);			
	}
}
__global__ void cuda_FracFDCoef(cufftComplex *d_wk1, cufftComplex *d_wk2, cufftComplex *d_wk3, float *d_diffcoef2, float alpha1, float alpha2, float alpha3, int Mx, int Mz, int in)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	int id = idz*Mx+idx;
	
	float dkx=2.0*pi/Mx;
	float dkz=2.0*pi/Mz;
	float kx,kz;
	float tempx;
	float tempz;
	int iN;
	
	if (idz < Mz && idx < Mx)
	{
		kx = idx*dkx;
		kz = idz*dkz;
		
		tempx = 0.0;
		tempz = 0.0;
				
		for (iN=1;iN<=in;iN++)
		{
		
			tempx += 4.0*d_diffcoef2[(in-1)*radius + iN-1]*sinf(0.5*iN*kx)*sinf(0.5*iN*kx);
			tempz += 4.0*d_diffcoef2[(in-1)*radius + iN-1]*sinf(0.5*iN*kz)*sinf(0.5*iN*kz);
		}
		
		d_wk1[id].x = powf(tempx+tempz, 0.5*alpha1);
		d_wk2[id].x = powf(tempx+tempz, 0.5*alpha2);
		d_wk3[id].x = powf(tempx+tempz, 0.5*alpha3);
		
		d_wk1[id].y = 0.0;
		d_wk2[id].y = 0.0;
		d_wk3[id].y = 0.0;
	}
}
__global__ void cudacpyw(cufftComplex *d_wk1, cufftComplex *d_wk2, cufftComplex *d_wk3, float *d_wt1, float *d_wt2, float *d_wt3, float *d_w1, float *d_w2, float *d_w3, int Mz, int Mx, int iN)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	int id = idz*Mx+idx;
	int Mxh = Mx/2;
	int Mzh = Mz/2;
	
	if (idz < Mz && idx < Mx)
	{
		if (idx < Mxh)
		{
			d_wt1[id] = d_wk1[idz*Mx+idx+Mxh+1].x/(Mx*Mz);
			d_wt2[id] = d_wk2[idz*Mx+idx+Mxh+1].x/(Mx*Mz);
			d_wt3[id] = d_wk3[idz*Mx+idx+Mxh+1].x/(Mx*Mz);	
		}
		if (idx >= Mxh)
		{
			d_wt1[id] = d_wk1[idz*Mx+idx-Mxh].x/(Mx*Mz);
			d_wt2[id] = d_wk2[idz*Mx+idx-Mxh].x/(Mx*Mz);
			d_wt3[id] = d_wk3[idz*Mx+idx-Mxh].x/(Mx*Mz);				
		}
	}
	__syncthreads();
		
	if (idz < Mz && idx < Mx)
	{
		if (idz < Mzh)
		{
			d_w1[id] = d_wt1[(idz+Mzh+1)*Mx+idx];
			d_w2[id] = d_wt2[(idz+Mzh+1)*Mx+idx];
			d_w3[id] = d_wt3[(idz+Mzh+1)*Mx+idx];	
		}
		if (idz >= Mzh)
		{
			d_w1[id] = d_wt1[(idz-Mzh)*Mx+idx];
			d_w2[id] = d_wt2[(idz-Mzh)*Mx+idx];
			d_w3[id] = d_wt3[(idz-Mzh)*Mx+idx];				
		}
	}	
	__syncthreads();	
}
//=========================================================================================================
__global__ void cuda_forward_p_iso(float *d_p2, float *d_p1, float *d_p0, float *d_pt, float *d_w1, float *d_w2, float *d_w3, float *d_vp, float *d_gm, int *d_norderx, int *d_norderz, 
				float alpha1, float alpha2, float alpha3, float *d_c1, float *d_c2, float *d_c3, float *d_c4, float dt, float dx, float dz, float fdom, int npml, int nxpad, int nzpad, int Mx, int Mz)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + Mz/2;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + Mx/2;	
	int idzl = threadIdx.x;
	int idxl = threadIdx.y;	
	int idzm = idz - Mz/2;
	int idxm = idx - Mx/2;	
	int tidz = idzl + Mz/2;
	int tidx = idxl + Mx/2;	
	
	int Nx = nxpad + Mx - 1;
	int id,idk,inx,inz;	
	int Mxh = Mx/2;
	int Mzh = Mz/2;
	int radiusx,radiusz,Radius;

	float c1,c2,c3,c4;
	float vp,gm,eta,tau;
	float temp1,temp2,temp3,temp4;
	float w0=100*pi*fdom; // 100
	
	__shared__ float p1[Block_Sizez + N][Block_Sizex + N];
	__shared__ float pt[Block_Sizez + N][Block_Sizex + N];
	
	
	if (idz >= Mzh && idz <= nzpad+Mzh-1 && 
	    idx >= Mxh && idx <= nxpad+Mxh-1)
	{
		id = idz*Nx+idx;
		d_pt[id] = (d_p1[id]-d_p0[id])/dt;
	}
	__syncthreads();	

	// four conner
	if (idxl <  Mxh && idzl <  Mzh){                
		p1[idzl][idxl]         = d_p1[(idz-Mzh)*Nx + idx - Mxh];
		pt[idzl][idxl]         = d_pt[(idz-Mzh)*Nx + idx - Mxh];
	}	
	if (idxl <  Mxh && idzl >= blockDim.x - Mzh){                
		p1[idzl + Mz-1][idxl]         = d_p1[(idz+Mzh)*Nx + idx - Mxh];
		pt[idzl + Mz-1][idxl]         = d_pt[(idz+Mzh)*Nx + idx - Mxh];
	}
	if (idxl >= blockDim.y - Mxh && idzl <  Mzh){                
		p1[idzl][idxl + Mx-1]         = d_p1[(idz-Mzh)*Nx + idx + Mxh];
		pt[idzl][idxl + Mx-1]         = d_pt[(idz-Mzh)*Nx + idx + Mxh];
	}
	if (idxl >= blockDim.y - Mxh && idzl >= blockDim.x - Mzh){                
		p1[idzl + Mz-1][idxl + Mx-1]         = d_p1[(idz+Mzh)*Nx + idx + Mxh];
		pt[idzl + Mz-1][idxl + Mx-1]         = d_pt[(idz+Mzh)*Nx + idx + Mxh];
	}
	// four boundary
	if (idxl <  Mxh){                
		p1[tidz][idxl]         = d_p1[idz*Nx + idx - Mxh];
		pt[tidz][idxl]         = d_pt[idz*Nx + idx - Mxh];
	}
	if (idxl >= blockDim.y - Mxh){
		p1[tidz][idxl + Mx-1]     = d_p1[idz*Nx + idx + Mxh];
		pt[tidz][idxl + Mx-1]     = d_pt[idz*Nx + idx + Mxh];
	}
	if (idzl <  Mzh){
		p1[idzl][tidx]         = d_p1[(idz - Mzh)*Nx + idx];
		pt[idzl][tidx]         = d_pt[(idz - Mzh)*Nx + idx];
	}
	if (idzl >= blockDim.x - Mzh){     
		p1[idzl + Mz-1][tidx]     = d_p1[(idz + Mzh)*Nx + idx];
		pt[idzl + Mz-1][tidx]     = d_pt[(idz + Mzh)*Nx + idx];
	}

	p1[tidz][tidx] = d_p1[idz*Nx + idx];
	pt[tidz][tidx] = d_pt[idz*Nx + idx];
	__syncthreads();	

	if (idz > Mzh && idz < nzpad+Mzh-1 && 
	    idx > Mxh && idx < nxpad+Mxh-1)
	{
		id=idz*Nx+idx;
		idk= (idz-radius)*nxpad+(idx-radius);
		
		vp = d_vp[id];
		gm = d_gm[id];	
		
		eta = -powf(vp,gm)*powf(w0,-2.0*gm)*cosf(pi*gm);
		tau = -powf(vp,gm-0.5)*powf(w0,-2.0*gm)*sinf(pi*gm);	

		c1 = d_c1[idk];
		c2 = d_c2[idk];
		c3 = d_c3[idk];
		c4 = d_c4[idk];	
		
		radiusx = d_norderx[idxm];
		radiusz = d_norderz[idzm];  
		
		Radius = radiusx<radiusz?radiusx:radiusz;
		
		temp1 = 0.0;
		temp2 = 0.0;
		temp3 = 0.0;
		temp4 = 0.0;

		for (inz = -Radius; inz <= Radius; inz++)
		for (inx = -Radius; inx <= Radius; inx++)
		{ 
			temp1 += d_w1[(Radius-1)*Mx*Mz + (inz+Mzh)*Mx + (inx+Mxh)]*p1[tidz+inz][tidx+inx];
			temp2 += d_w2[(Radius-1)*Mx*Mz + (inz+Mzh)*Mx + (inx+Mxh)]*p1[tidz+inz][tidx+inx]; 
			
			temp3 += d_w3[(Radius-1)*Mx*Mz + (inz+Mzh)*Mx + (inx+Mxh)]*pt[tidz+inz][tidx+inx];
			temp4 += d_w1[(Radius-1)*Mx*Mz + (inz+Mzh)*Mx + (inx+Mxh)]*pt[tidz+inz][tidx+inx];
        }
		temp1 /= powf(dx,alpha1);
		temp2 /= powf(dx,alpha2);
		
		temp3 /= powf(dx,alpha3); 
		temp4 /= powf(dx,alpha1);         		
		
		d_p2[id] =  2.0*d_p1[id] - d_p0[id] + dt*dt*vp*cosf(0.5*pi*gm)*cosf(0.5*pi*gm)*(eta*((1.0+c3*gm)*temp1 + c4*gm*temp2) 
																					  + tau*((1.0+c1*gm)*temp3 + c2*gm*temp4));
	}
}
__global__ void cuda_abc(float *d_p2, float *d_p1, float *d_p0, float *d_vp, int nxpad, int nzpad, int Mx, int Mz, int npml, float dx, float dz, float dt)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + Mz/2;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + Mx/2;
	int Mxh = Mx/2;
	int Mzh = Mz/2;
	int Nx = nxpad + Mx-1;
	
	int id=idz*Nx+idx;

	float w,s,t1,t2,t3;
	
	// left ABC ...
	if(idx>=Mxh && idx<Mxh+npml &&
	   idz>=Mzh && idz<Mzh+nzpad )
	{
		w=(float)(idx-Mxh)/npml;
		
		s=sqrtf(d_vp[id])*dt/dx;
		t1=(2.0-s)*(1.0-s)/2.0;
		t2=s*(2.0-s);
		t3=s*(s-1.0)/2.0;
		
		d_p2[id]=(1.0-w)*(2.0*(t1*d_p1[id]+t2*d_p1[id+1]+t3*d_p1[id+2])-(t1*t1*d_p0[id]+2.0*t1*t2*d_p0[id+1]+(2.0*t1*t3+t2*t2)*d_p0[id+2]+2.0*t2*t3*d_p0[id+3]+t3*t3*d_p0[id+4])) 
			     + w*d_p2[id];
	}

	// right ABC ...
	if(idx>=Mxh+nxpad-npml && idx<Mxh+nxpad &&
	   idz>=Mzh            && idz<Mzh+nzpad )
	{
		w=(float)(Mxh+nxpad-1-idx)/npml;
		
		s=sqrtf(d_vp[id])*dt/dx;
		t1=(2.0-s)*(1.0-s)/2.0;
		t2=s*(2.0-s);
		t3=s*(s-1.0)/2.0;
		
		d_p2[id]=(1.0-w)*(2.0*(t1*d_p1[id]+t2*d_p1[id-1]+t3*d_p1[id-2])-(t1*t1*d_p0[id]+2.0*t1*t2*d_p0[id-1]+(2.0*t1*t3+t2*t2)*d_p0[id-2]+2.0*t2*t3*d_p0[id-3]+t3*t3*d_p0[id-4])) 
			     + w*d_p2[id];
	}


	// up ABC ...
	if(idz>=Mzh && idz<Mzh+npml &&
	   idx>=Mxh && idx<Mxh+nxpad )
	{
		w=(float)(idz-Mzh)/npml;
		
		s=sqrtf(d_vp[id])*dt/dz;
		t1=(2.0-s)*(1.0-s)/2.0;
		t2=s*(2.0-s);
		t3=s*(s-1.0)/2.0;
		
		d_p2[id]= (1.0-w)*(2.0*(t1*d_p1[id]+t2*d_p1[id+Nx]+t3*d_p1[id+2*Nx])-(t1*t1*d_p0[id]+2.0*t1*t2*d_p0[id+Nx]+(2.0*t1*t3+t2*t2)*d_p0[id+2*Nx]+2.0*t2*t3*d_p0[id+3*Nx]+t3*t3*d_p0[id+4*Nx]))
			      + w*d_p2[id];			
	}

	// bottom ABC ...
	if(idz>=Mzh+nzpad-npml && idz<Mzh+nzpad &&
	   idx>=Mxh            && idx<Mxh+nxpad )
	{
		w=(float)(Mzh+nzpad-1-idz)/npml;
		
		s=sqrtf(d_vp[id])*dt/dz;
		t1=(2.0-s)*(1.0-s)/2.0;
		t2=s*(2.0-s);
		t3=s*(s-1.0)/2.0;

		d_p2[id]= (1.0-w)*(2.0*(t1*d_p1[id]+t2*d_p1[id-Nx]+t3*d_p1[id-2*Nx])-(t1*t1*d_p0[id]+2.0*t1*t2*d_p0[id-Nx]+(2.0*t1*t3+t2*t2)*d_p0[id-2*Nx]+2.0*t2*t3*d_p0[id-3*Nx]+t3*t3*d_p0[id-4*Nx]))
			      + w*d_p2[id];		
	}
	__syncthreads();	
}
__global__ void cuda_backward_p_tti(float *d_p2, float *d_p1, float *d_p0, float *d_pdx, float *d_pdz, 
				    float *d_vp, float *d_ep, float *d_de, float *d_th, float *d_diffcoef1, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + radius;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + radius;
	
	int idzl = threadIdx.x;
	int idxl = threadIdx.y;
	
	int idzm = idz-radius;
	int idxm = idx-radius;

	int tidz  =  idzl + radius;
	int tidx  =  idxl + radius;
	
	int id;
	
	int Nx = nxpad + N;

	float diffx1,diffz1,
              diffx2,diffz2,diffxz;
      	      
        float vp,ep,de,th;
        
        int radiusx,radiusz;
        int inx,inz;

	float gradx,gradz,grada;
	float S1,S2,S3,S;
	float pc1,pc2,pc3;

	__shared__ float pp[Block_Sizez + N][Block_Sizex + N];
	__shared__ float px[Block_Sizez + N][Block_Sizex];

	if (idxl <  radius)                    pp[tidz][idxl]         = d_p1[idz*Nx + idx - radius];
	if (idxl >= blockDim.y - radius)       pp[tidz][idxl + N]     = d_p1[idz*Nx + idx + radius];
	if (idzl <  radius) 
	{
				               pp[idzl][tidx]         = d_p1[(idz - radius)*Nx + idx];
				               px[idzl][idxl]         = d_pdx[(idz - radius)*Nx + idx];
	}
	if (idzl >= blockDim.x - radius)
	{
					       pp[idzl + N][tidx]     = d_p1[(idz + radius)*Nx + idx];
					       px[idzl + N][idxl]     = d_pdx[(idz + radius)*Nx + idx];
	}
	pp[tidz][tidx] = d_p1[idz*Nx + idx];
	px[tidz][idxl] = d_pdx[idz*Nx + idx];
	__syncthreads();

	if (idz >= radius+npml && idz < nzpad+radius-npml &&
	    idx >= radius+npml && idx < nxpad+radius-npml )
	{
		id = idz*Nx+idx;
		
            	diffx1 = 0.0;
            	diffz1 = 0.0;
            
            	diffx2 = 0.0;
            	diffz2 = 0.0;
            	diffxz = 0.0;
            	  
		vp = d_vp[id];   
            	ep = d_ep[id];
            	de = d_de[id];
            	th = d_th[id];           	
           
            	pc1 = (1.0+2.0*ep)*cos(th)*cos(th) + sin(th)*sin(th);
	    	pc2 =-2.0*ep*sin(2.0*th); 
	    	pc3 = (1.0+2.0*ep)*sin(th)*sin(th) + cos(th)*cos(th);          
		
		radiusx = d_norderx[idxm];
            	radiusz = d_norderz[idzm];            
            
		for (inx = 1; inx <= radiusx; inx++){ 
			diffx2 += d_diffcoef2[(radiusx-1)*radius + inx - 1]*(pp[tidz][tidx + inx] + pp[tidz][tidx - inx] - 2.0*pp[tidz][tidx]);
                }
		for (inz = 1; inz <= radiusz; inz++){ 
			diffxz += d_diffcoef1[(radiusz-1)*radius + inz - 1]*(px[tidz + inz][idxl] - px[tidz - inz][idxl]);     
			diffz2 += d_diffcoef2[(radiusz-1)*radius + inz - 1]*(pp[tidz + inz][tidx] + pp[tidz - inz][tidx] - 2.0*pp[tidz][tidx]); 
		}		
		diffx2 = diffx2/dx/dx;
		diffz2 = diffz2/dz/dz;
		diffxz = diffxz/dz;
		
		diffx1 = d_pdx[idz*Nx+idx];
		diffz1 = d_pdz[idz*Nx+idx];		
		
		gradx = cos(th)*diffx1 + sin(th)*diffz1;
		gradz =-sin(th)*diffx1 + cos(th)*diffz1;
				
		grada = sqrtf(gradx*gradx+gradz*gradz);
			
		gradx = (!isnan(gradx/grada) && !isinf(gradx/grada)) ? (gradx/grada):0.0;
		gradz = (!isnan(gradz/grada) && !isinf(gradz/grada)) ? (gradz/grada):0.0;		
			
		S1 = 8.0*(ep-de)*(gradx*gradx*gradz*gradz);
		S2 = gradz*gradz+(1.0+2.0*ep)*(gradx*gradx);
		S2 *= S2;
		
		S3 = (!isnan(S1/S2) && !isinf(S1/S2)) ? (S1/S2):0.0;
		S = 0.5*(1.0 + sqrtf(1.0-S3));
		
	
	/*	S1 = 8.0*(ep-de)*(gradx*gradx*gradz*gradz);
		S2 = 1.0+2.0*ep*(gradx*gradx);
		S2 *= S2;
		S = 0.5*(1.0 + sqrtf(1.0-S1/S2));
	*/	
		d_p2[id] =  2.0*d_p1[id] - d_p0[id] + dt*dt*vp*S*(pc1*diffx2 + pc2*diffxz + pc3*diffz2);
	}	
}
__global__ void cuda_backward_p_vti(float *d_p2, float *d_p1, float *d_p0, 
				    float *d_vp, float *d_ep, float *d_de, float *d_diffcoef1, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + radius;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + radius;
	int idzl = threadIdx.x;
	int idxl = threadIdx.y;
	int idzm = idz-radius;
	int idxm = idx-radius;
	
	int tidz  =  idzl + radius;
	int tidx  =  idxl + radius;
	
	int id;
	
	int Nx = nxpad + N;

	float diffx1,diffz1,diffx2,diffz2;
	float vp,ep,de;
	
	float gradx,gradz,grada;
	float S1,S2,S3,S;
	float Cx,Cz;
	
	int radiusx,radiusz;
	int inx,inz;

	__shared__ float p[Block_Sizez + N][Block_Sizex + N];

	if (idxl <  radius)                    p[tidz][idxl]         = d_p1[idz*Nx + idx - radius];
	if (idxl >= blockDim.y - radius)       p[tidz][idxl + N]     = d_p1[idz*Nx + idx + radius];
	if (idzl <  radius)                    p[idzl][tidx]         = d_p1[(idz - radius)*Nx + idx];
	if (idzl >= blockDim.x - radius)       p[idzl + N][tidx]     = d_p1[(idz + radius)*Nx + idx];

	p[tidz][tidx] = d_p1[idz*Nx + idx];
	__syncthreads();

	if (idz >= radius+npml && idz < nzpad+radius-npml &&
	    idx >= radius+npml && idx < nxpad+radius-npml )
	{
		id = idz*Nx + idx;
		
		diffx1 = 0.0;
		diffz1 = 0.0;
		
		diffx2 = 0.0;
		diffz2 = 0.0;
            
		radiusx = d_norderx[idxm];
		radiusz = d_norderz[idzm];
            
		vp = d_vp[id];
		ep = d_ep[id];
		de = d_de[id];

		Cx = 1.0+2.0*ep;
		Cz = 1.0;
            
		for (inx = 1; inx <= radiusx; inx++){ 
			diffx1 += d_diffcoef1[(radiusx-1)*radius + inx - 1]*(p[tidz][tidx + inx] - p[tidz][tidx - inx]);
                	diffx2 += d_diffcoef2[(radiusx-1)*radius + inx - 1]*(p[tidz][tidx + inx] + p[tidz][tidx - inx] - 2.0*p[tidz][tidx]);
                }
		for (inz = 1; inz <= radiusz; inz++){ 
			diffz1 += d_diffcoef1[(radiusz-1)*radius + inz - 1]*(p[tidz + inz][tidx] - p[tidz - inz][tidx]);     
			diffz2 += d_diffcoef2[(radiusz-1)*radius + inz - 1]*(p[tidz + inz][tidx] + p[tidz - inz][tidx] - 2.0*p[tidz][tidx]); 
		}
		
		diffx1 = diffx1/dx;
		diffz1 = diffz1/dz;
		diffx2 = diffx2/dx/dx;
		diffz2 = diffz2/dz/dz;
		
		gradx = diffx1;
		gradz = diffz1;
		grada = sqrtf(gradx*gradx+gradz*gradz);
			
		gradx = (!isnan(gradx/grada) && !isinf(gradx/grada)) ? (gradx/grada):0.0;
		gradz = (!isnan(gradz/grada) && !isinf(gradz/grada)) ? (gradz/grada):0.0;		
			
		S1 = 8.0*(ep-de)*(gradx*gradx*gradz*gradz);
		S2 = gradz*gradz+(1.0+2.0*ep)*(gradx*gradx);
		S2 *= S2;
		
		S3 = (!isnan(S1/S2) && !isinf(S1/S2)) ? (S1/S2):0.0;
		S = 0.5*(1.0 + sqrtf(1.0-S3));

		d_p2[id] =  2.0*d_p1[id] - d_p0[id] + dt*dt*vp*S*(Cx*diffx2 + Cz*diffz2);
	}
}
__global__ void cuda_backward_p_iso(float *d_p2, float *d_p1, float *d_p0, 
				    float *d_vp, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + radius;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + radius;
	int idzl = threadIdx.x;
	int idxl = threadIdx.y;
	int idzm = idz-radius;
	int idxm = idx-radius;
	
	int tidz  =  idzl + radius;
	int tidx  =  idxl + radius;
	
	int id;
	
	int Nx = nxpad + N;

	float diffx2,diffz2;
	float vp;
	
	int radiusx,radiusz;
	int inx,inz;

	__shared__ float p[Block_Sizez + N][Block_Sizex + N];

	if (idxl <  radius)                    p[tidz][idxl]         = d_p1[idz*Nx + idx - radius];
	if (idxl >= blockDim.y - radius)       p[tidz][idxl + N]     = d_p1[idz*Nx + idx + radius];
	if (idzl <  radius)                    p[idzl][tidx]         = d_p1[(idz - radius)*Nx + idx];
	if (idzl >= blockDim.x - radius)       p[idzl + N][tidx]     = d_p1[(idz + radius)*Nx + idx];

	p[tidz][tidx] = d_p1[idz*Nx + idx];
	__syncthreads();

	if (idz >= radius+npml && idz < nzpad+radius-npml &&
	    idx >= radius+npml && idx < nxpad+radius-npml )
	{
		id = idz*Nx + idx;
		
		diffx2 = 0.0;
		diffz2 = 0.0;
            
		radiusx = d_norderx[idxm];
		radiusz = d_norderz[idzm];
            
		vp = d_vp[id];
            
		for (inx = 1; inx <= radiusx; inx++)
			diffx2 += d_diffcoef2[(radiusx-1)*radius + inx - 1]*(p[tidz][tidx + inx] + p[tidz][tidx - inx] - 2.0*p[tidz][tidx]);
		for (inz = 1; inz <= radiusz; inz++) 
			diffz2 += d_diffcoef2[(radiusz-1)*radius + inz - 1]*(p[tidz + inz][tidx] + p[tidz - inz][tidx] - 2.0*p[tidz][tidx]); 
		
		diffx2 = diffx2/dx/dx;
		diffz2 = diffz2/dz/dz;
		
		d_p2[id] =  2.0*d_p1[id] - d_p0[id] + dt*dt*vp*(diffx2 + diffz2);
	}
}
//=========================================================================================================
__global__ void save_pmllr_f(float *d_p, float *d_pmlplr, int nxlength, int nzlength, int npml, int Ntemp)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	
	int Nx=nxlength+N;
	int Radius = Ntemp/2;
	int id;
	int nl = Radius*(nzlength-2*npml);
	if (idz < nzlength-2*npml)
	{
		if (blockIdx.y == 0)  // left
		{
			id = idz*Radius + idx;
			if (idx < Radius)
				d_pmlplr[id] = d_p[(idz+npml+radius)*Nx + npml+radius+idx-Radius];
		}
		if (blockIdx.y == 1) // right
		{
			id = idz*Radius + idx-Radius;
			if (idx < Ntemp)
				d_pmlplr[id+nl] = d_p[(idz+npml+radius)*Nx + nxlength+radius-npml+idx-Radius];	
		}
	}	
}
__global__ void save_pmltb_f(float *d_p, float *d_pmlptb, int nxlength, int nzlength, int npml, int Ntemp)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	
	int Nx=nxlength+N;
	int Radius = Ntemp/2;
	int id;
	int nl = Radius*(nxlength-2*npml);
	if (idx < nxlength-2*npml)
	{
		if (blockIdx.x == 0)  // top
		{
			id = idz*(nxlength-2*npml) + idx;
			if (idz < Radius)
				d_pmlptb[id] = d_p[(idz+npml+radius-Radius)*Nx + idx+npml+radius];
		}
		if (blockIdx.x == 1) // bottom
		{
			id = (idz-Radius)*(nxlength-2*npml) + idx;
			if (idz < Ntemp)
				d_pmlptb[id+nl] = d_p[(nzlength+radius-npml+idz-Radius)*Nx + idx+npml+radius];	
		}
	}	
}
__global__ void read_pmllr_f(float *d_p, float *d_pmlplr, int nxlength, int nzlength, int npml, int Ntemp)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	
	int Nx=nxlength+N;
	int Radius = Ntemp/2;
	int id;
	int nl = Radius*(nzlength-2*npml);
	if (idz < nzlength-2*npml)
	{
		if (blockIdx.y == 0)  // left
		{
			id = idz*Radius + idx;
			if (idx < Radius)
				d_p[(idz+npml+radius)*Nx + npml+radius+idx-Radius]  = d_pmlplr[id];
		}
		if (blockIdx.y == 1) // right
		{
			id = idz*Radius + idx-Radius;
			if (idx < Ntemp)
				d_p[(idz+npml+radius)*Nx + nxlength+radius-npml+idx-Radius]  = d_pmlplr[id+nl];	
		}
	}	
}
__global__ void read_pmltb_f(float *d_p, float *d_pmlptb, int nxlength, int nzlength, int npml, int Ntemp)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = threadIdx.y + blockDim.y * blockIdx.y;
	
	int Nx=nxlength+N;
	int Radius = Ntemp/2;
	int id;
	int nl = Radius*(nxlength-2*npml);
	if (idx < nxlength-2*npml)
	{
		if (blockIdx.x == 0)  // top
		{
			id = idz*(nxlength-2*npml) + idx;
			if (idz < Radius)
				d_p[(idz+npml+radius-Radius)*Nx + idx+npml+radius]  = d_pmlptb[id];
		}
		if (blockIdx.x == 1) // bottom
		{
			id = (idz-Radius)*(nxlength-2*npml) + idx;
			if (idz < Ntemp)
				d_p[(nzlength+radius-npml+idz-Radius)*Nx + idx+npml+radius]  = d_pmlptb[id+nl];	
		}
	}	
}
//=========================================================================================================
__global__ void cuda_imagingconditon_lsrtm(float *d_ps2, float *d_ps1, float *d_ps0, float *d_pr1,float *d_image, float *d_illum, float dt, int nxpad, int nzpad, int npml)
{
	int idz = threadIdx.x + blockDim.x * blockIdx.x + radius;
	int idx = threadIdx.y + blockDim.y * blockIdx.y + radius;

	int idf,idh;

	if (idz >= radius+npml && idz < nzpad+radius-npml && 
	    idx >= radius+npml && idx < nxpad+radius-npml)
	{
		idf = idz*(nxpad+N)+idx;
		idh = (idz-radius-npml)*(nxpad-2*npml)+idx-radius-npml;
		
		d_image[idh] -= d_pr1[idf]*(d_ps2[idf] + d_ps0[idf] - 2.0*d_ps1[idf])/dt/dt;
		d_illum[idh] += d_ps1[idf]*d_ps1[idf];
	}	
}
//=========================================================================================================
void check_gpu_error(const char *msg)
// < check GPU errors >
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("Cuda error: %s: %s",msg,cudaGetErrorString(err));
		exit(0);
	}
}
