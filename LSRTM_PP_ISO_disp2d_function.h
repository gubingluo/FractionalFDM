#ifndef FUNCTION_H
#define FUNCTION_H

int npfar (int nmin);
float sum1part(float *diffcoef, int n);
double sum(double *data, int n);
float sum2(float **data, int nx, int nz);
void arrayabs(float *data1, float *data, int n);
float ***Creat3dArray(int m, int n, int k);
void free3dArray(float ***tt, int m, int n, int k);
float **Creat2dArray(int m, int n);
void free2dArray(float **tt, int m, int n);
int **Creat2dArray_int(int m, int n);
void free2dArray_int(int **tt, int m, int n);
void fmemset1(float *p, int len);
void fmemset2(float **p, int nz, int nx);
void fmemset3(float ***p, int nz, int nx, int nt);
void fmemcpy3(float ***dest, float ***sour, int nz, int nx, int nt);
void fmemset1v(float *p, int len, float v);
void fmemset2v(float **p, int nz, int nx, float v);
void fmemset1vp(float *p, int len, float *vp);
void fmemset2vp(float **p, int nz, int nx, float **vp);
float Maxval2(float **v, int nz, int nx);
float Minval2(float **v, int nz, int nx);
float Maxval1(float *v, int n);
float Minval1(float *v, int n);
float absMaxval2(float **v, int nz, int nx);
float absMinval2(float **v, int nz, int nx);
float absMaxval3(float ***v, int nz, int nx, int nt);
float absMinval3(float ***v, int nz, int nx, int nt);
float absMaxval2_AB(float **A, float **B, int nz, int nx);
float absMaxval1(float *v, int n);
float absMinval1(float *v, int n);
void MatOperSca1(float *Matrix, float Scalar, int Operation, int n);
void MatOperSca2(float **Matrix, float Scalar, int Operation, int nz, int nx);
void Where1(float *Matrix, float Scalar, int Operation, int n);
void Where2(float **Matrix, float Scalar, int Operation, int nz, int nx);
float multisum1(float *p, float signp, float *q, float signq, int len);
float multisum3(float ***p, float signp, float ***q, float signq, int nz, int nx, int nshot);
void sum2d(float **temp, int m, int n, float sum);
float sumabs(float *data, int n);
float Maxval(float *v, int nxz);
float Minval(float *v, int nxz);
float absMaxval(float *v, int nxz);
float absMinval(float *v, int nxz);
void fmemset(float *input, float value, int nxz);
void ReadVelMod(char velfile[40], int nx ,int nz ,float *vp);
void Diff_coeff1_displacement(float *data, int N_order);
void Diff_coeff2_displacement(float *data, int N_order);
void Diff_coeff2_displacement_LS(float *data, int N_order, double khmax, double alpha);
void Outputimage(float **record, int nz, int nx, float dx, char buff[40], int Out_flag);
void Outputrecord(float *record, int nt, int nx, float dt, char buff[40], int Out_flag);
void Inputrecord(float *record, int nt, int nx, char buff[40],int In_flag);
void Velsmooth(float *vp, int nx, int nz, int npml);
void check_grid_sanity(int N_order, float *diffcoef, float vpmax, float vpmin, float fdom, float dx, float dz, float dt);
void Searchmaxgrad(int nx, int nz, int npml, float *Grad, float *maxgrad, int *nxgrad, int *nzgrad);
void UpdateImage(int nx, int nz, int npml, float *Image, float *d, float alpha);
void UpdateImage3d(int nz, int nx, int nshot, int startshot, int dshot, float ***Image, float ***d, double alpha);
void Updateodcig3d(int nz, int nx, int noffset, int npml, float ***Image, float ***d, double alpha);
void sincInterp(int nx, int nt0, int nt, float dt0, float dt, float *st0, float *st, float **seis0, float **seis);
void scaleillum_grad(float *illum, float *grad, int n, float damp);
void scale_gradient(float *grad, float *illum, int nz, int nx, int npml);
void scaled2(float ***d, int nz, int nx, int nshot, int startshot, int dshot);
void scale_image(float *grad, float *illum, int nz, int nx, int npml);
void scale_image1(float *grad, float *illumS, float *illumR, int nz, int nx, int npml);
void gradtaperv(int nx, int nz, int npml, int npos, float *g);
void gradtaperh1(int nx, int nz, int npml, int npos, float **g);
void gradtaperh(int nx, int nz, int npml, int npos, float *g);
void imagetaperh(int nx, int nz, int npml, int npos, float **image);
void gradtaper(float **g, int nx, int nz, int npml, float dx, float dz, float spx, float spz);
float Cal_epsilon(float *drecord, float n);
void velsmooth(int nx, int nz, float dx, float dz, float *vel);
void Velsmooth(float **vp, int nx, int nz, int npml);
void vpsmooth(float *vp,int n1,int n2,int nsp);

void pmlvelsmooth1d(float *vp, int nx, int nz, int npml);
float Velmaxpml1d(float *vp, int nz, int nx, int npml);
void Order_position_2order(int *nOrderx, int *nOrderz, int npml, int nx, int nz, int Ntemp);
void PML_Coeff_2order(float *ddx1, float *ddz1, float *ddx2, float *ddz2, float vpmax, int npml, int nx, int nz, float dt, float dx, float dz);
void Output1d(float *record, int nt, int nx, float dt, char *buff, int Out_flag);
void Input1d(float *record, int nt, int nx, char *buff,int In_flag);
void Output2d(float **record, int nt, int nx, float dt, char *buff, int Out_flag);
void Input2d(float **record, int nt, int nx, char *buff,int In_flag);
void Output3d(float ***record, int nt, int nx, int nh, float dt, char *buff, int Out_flag);
void Input3d(float ***record, int nt, int nx, int nh, char *buff,int In_flag);
void Output1D(float *record, int nt, int nx, float dt, char *buff, int Out_flag, int par_flag, int fldr, float sx0, float sz0, float gz0, float gx0, float dgx0, float offsx1, float offsx2, float DX);
void mute1x(float *seisobs, float *vp, float spx, float spz, int nt, int nx, int npml, int nw, int tlength, float dx, float dz, float dt);
void imagetaperh3d(int nx, int nz, int npml, int nshot, int startshot, int dshot, int nwidth, float damp, float ***g);
void odcigtaperh3d(int nx, int nz, int npml, int noffset, int nwidth, float damp, float ***odcigs);


void rmNaN1d(float *Image, float value, int n);
void rmNaN2d(float **Image, float value, int n1, int n2);
void rmNaN3d(float ***Image, float value, int n1, int n2, int n3);

void slantstack(float ***adcigs, float ***odcigs, float *p, int noffset, int ntheta, int nz, int nx, float dx ,float dz);
void cigregu(float ***adcigs, int nz,int nx, int np, int nh, int nsp);
void cigsmth(float ***adcigs, int nz,int nx, int np, int nsp);
void cignorm(float ***adcigs, int nz, int nx, int np);
int band(float *b,float *d, int n, int l, int il, int m);

#endif
