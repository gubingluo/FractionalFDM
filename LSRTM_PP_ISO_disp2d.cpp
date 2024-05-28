//////////////////////////////////////////////////////////////////////////
//      2D finite difference time domain acoustic wave 
// two-order displacement wave equation forward simulation
//   multi-shots for least square reverse time migration
//   all shots must share the same receiver
//   modified: 2015-10-07
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include "LSRTM_PP_ISO_disp2d_function.h"
#include "common.h"

extern "C" int gpunormforward1_iso(int idevice, int nt, int nx, int nz, int sshotx, int nshotx, int dshotx, int sshotz, int nshotz, int dshotz, int NX, int npml, int imedia, int Ntemp, int ihomo,
				float dt, float dx, float dz, float fdom, float amp, float alp, float spx0, float spz0, float dspx, float dspz, float *vp0, float *Qp0);

void interpvel1(float *vp0, float *vp, int nx0, int nz0, int nx, int nz, int lx, int lz);

int main(int argc, char* argv[])
{
	float dt,xmax,zmax,dx,dz,dx0,dz0,fdom,amp,alp,spx0,spz0,dspx,dspz,threshold,direct,direct0,gd;

	int nt,ndt,sshotx,nshotx,dshotx,sshotz,nshotz,dshotz,ntsnap,npml,nx,nz,nx0,nz0,NX,objflag,nw,tlength,nwinth,lx,lz;
	int iter,maxiter,rbell,irecord,isnap,iborn,ismth,idevice,imedia,Ntemp,iq,ihomo,cflag;
	char buff[100],buffimage[40],vp_file[40],Qp_file[40];
	char cmd[40],cmd1[100];
	int i,ishot,ix,iz;
	clock_t start,stop;
	int err;

	FILE *fp=NULL;

	if ((fp=fopen("./input/cmd.in","r"))==NULL)
	{
		printf("cmd.in is not exist !\n");
		exit(0);
	}
	fgets(buff,100,fp);
	fscanf(fp,"%s", cmd);
	fclose(fp);

	sprintf(cmd1,"./input/%s",cmd);

	if ((fp=fopen(cmd1,"r"))==NULL)
	{
		printf("%s is not exist !\n",cmd);
		exit(0);
	}
	fgets(buff,100,fp);
	fscanf(fp,"%f", &dt);fscanf(fp,"%d",&nt);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%f", &xmax);fscanf(fp,"%f", &zmax);fscanf(fp,"%f", &dx0);fscanf(fp,"%f", &dz0);	
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%f", &fdom);fscanf(fp,"%f", &amp);fscanf(fp,"%f", &alp);	
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%f", &spx0);fscanf(fp,"%f", &spz0);fscanf(fp,"%f", &dspx); fscanf(fp,"%f", &dspz);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &sshotx);fscanf(fp,"%d", &sshotz);
	fscanf(fp,"%d", &nshotx);fscanf(fp,"%d", &nshotz);
	fscanf(fp,"%d", &dshotx);fscanf(fp,"%d", &dshotz);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &NX);fscanf(fp,"%d", &Ntemp);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &npml);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &imedia);fscanf(fp,"%d", &ismth);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &irecord);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &idevice);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%d", &ihomo);fscanf(fp,"%d", &iq);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%s", vp_file);
	fgets(buff,100,fp);fgets(buff,100,fp);
	fscanf(fp,"%s", Qp_file);		
	fclose(fp);
	nx0=(int)(xmax/dx0)+1;
	nz0=(int)(zmax/dz0)+1;
	if (spx0<0 || spx0+(nshotx-1)*dspx>(nx0-1)*dx0 ||spx0+(nshotx-1)*dspx<0 ||
		spz0<0 || spz0+(nshotz-1)*dspz>(nz0-1)*dz0 ||spz0+(nshotz-1)*dspz<0)
	{
		printf("the shot position out of the model !\nplease check the parameter !\n");
		exit(0);
	}
	nx0=(int)(xmax/dx0)+1;
	nz0=(int)(zmax/dz0)+1;
	
	float *vp0=(float *)malloc((nx0*nz0)*sizeof(float)),
	      *Qp0=(float *)malloc((nx0*nz0)*sizeof(float));
	
	ReadVelMod(vp_file,nx0,nz0,vp0);
	if (iq == 1)
		ReadVelMod(Qp_file,nx0,nz0,Qp0);
	else
		for (ix=0;ix<nx0*nz0;ix++)
			Qp0[ix] = 8.5*powf(vp0[ix]/1000.0,2.2);
	
	if (ismth != 0)
		vpsmooth(Qp0,nz0,nx0,ismth);
	
	Output1D(vp0, nz0, nx0, dz0, "./input/myModel/Model_vp.su", 1, 0, 0, 0.0, 0.0, 0.0, 0.0, dx0, 0.0, 0.0, dx0);
	Output1D(Qp0, nz0, nx0, dz0, "./input/myModel/Model_Qp.su", 1, 0, 0, 0.0, 0.0, 0.0, 0.0, dx0, 0.0, 0.0, dx0);
	
	lx = 1;
	lz = 1;
	NX = (NX-1)*lx+1;
	
	nx = (nx0-1)*lx+1;
	nz = (nz0-1)*lz+1;
	
	dx = dx0/lx;
	dz = dz0/lz;
		
	float *vp=(float *)malloc((nx*nz)*sizeof(float)),
	      *Qp=(float *)malloc((nx*nz)*sizeof(float));
	
	
	interpvel1(vp0, vp, nx0, nz0, nx, nz, lx, lz);
	interpvel1(Qp0, Qp, nx0, nz0, nx, nz, lx, lz);
		
	free(vp0);
	free(Qp0);
	//======================================================================================//
	//===================main program=======================================================//
	//======================================================================================//
	printf("====================================================================\n");
	printf("====================================================================\n");
	printf("\nThe input file name is: %s\n\n",cmd);
	printf("================Calculate observation seismic record================\n");
	start = clock();
	if (irecord == 0)
	{
		err = gpunormforward1_iso(idevice,nt,nx,nz,sshotx,nshotx,dshotx,sshotz,nshotz,dshotz,NX,npml,imedia,Ntemp,ihomo,
		  		dt,dx,dz,fdom,amp,alp,spx0,spz0,dspx,dspz,vp,Qp);
		printf("================forward modeling seismic record================\n");
	}
	stop = clock();
	printf("Time cost: %f (s)\n",((float)(stop-start))/CLOCKS_PER_SEC);

	free(vp);
	free(Qp);

	printf("The program calculate over !\n");
	return 0;
}
void interpvel1(float *vp0, float *vp, int nx0, int nz0, int nx, int nz, int lx, int lz)
{
	int ix,iz,il;
	float *vel=(float *)malloc(nx*nz0*sizeof(float));
	// interpolate model along x-direction
	for (iz=0;iz<nz0;iz++)
	{
		for (ix=0;ix<nx0-1;ix++)
		{
			for (il=0;il<lx;il++)
			{
				vel[iz*nx+ix*lx+il] = ((lx-il)*vp0[iz*nx0+ix] + il*vp0[iz*nx0+ix+1])/lx;
			}
		}
		vel[iz*nx+(nx0-1)*lx] = vp0[iz*nx0+nx0-1];
	}
	// interpolate model along z-direction
	for (ix=0;ix<nx;ix++)
	{
		for (iz=0;iz<nz0-1;iz++)
		{
			for (il=0;il<lz;il++)
			{
				vp[(iz*lz+il)*nx+ix] = ((lz-il)*vel[iz*nx+ix] + il*vel[(iz+1)*nx+ix])/lz;
			}
		}
		vp[(nz0-1)*lz*nx+ix] = vel[(nz0-1)*nx+ix];
	}
	free(vel);
}
