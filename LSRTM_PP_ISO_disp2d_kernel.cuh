#ifndef FUNCTION_CUH
#define FUNCTION_CUH

//==============================================================================
__global__ void cuda_ricker_wavelet(float *d_wavelet, float fdom, float dt, int nt);
__global__ void cuda_source(float *d_source, int nsx, int nsz, int nxpad, int nzpad, int Mx, int Mz, float amp, float alp, float dx2, float dz2);
//==============================================================================
__global__ void cuda_add_source(float *d_p2, float *d_source, float *d_vp, float *d_wlt, float dt2, int add, int nxpad, int nzpad, int Mx, int Mz, int it);
//==============================================================================
__global__ void cuda_record(float *d_p, float *d_seis, int npml, int nxpad, int Mx, int Mz);
__global__ void cuda_insert_record(float *d_p, float *d_vp, float *d_seis, int npml, int nxpad, int Mx, int Mz, float dt2);
//==============================================================================
__global__ void cuda_C1ToC4(float *d_c1, float *d_c2, float *d_c3, float *d_c4, float *d_gm, float m, int nxpad, int nzpad);
__global__ void cuda_FracFDCoef(cufftComplex *d_wk1, cufftComplex *d_wk2, cufftComplex *d_wk3, float *d_diffcoef2, float alpha1, float alpha2, float alpha3, int Mx, int Mz, int iN);
__global__ void cudacpyw(cufftComplex *d_wk1, cufftComplex *d_wk2, cufftComplex *d_wk3, float *d_wt1, float *d_wt2, float *d_wt3, float *d_w1, float *d_w2, float *d_w3, int Mz, int Mx, int iN);
//==============================================================================
__global__ void cuda_forward_p_iso(float *d_p2, float *d_p1, float *d_p0, float *d_pt, float *d_w1, float *d_w2, float *d_w3, float *d_vp, float *d_gm, int *d_norderx, int *d_norderz, 
				float alpha1, float alpha2, float alpha3, float *d_c1, float *d_c2, float *d_c3, float *d_c4, float dt, float dx, float dz, float fdom, int npml, int nxpad, int nzpad, int Mx, int Mz);
__global__ void cuda_abc(float *d_p2, float *d_p1, float *d_p0, float *d_vp, int nxpad, int nzpad, int Mx, int Mz, int npml, float dx, float dz, float dt);
//==============================================================================
__global__ void cuda_backward_p_tti(float *d_p2, float *d_p1, float *d_p0, float *d_pdx, float *d_pdz, 
				    float *d_vp, float *d_ep, float *d_de, float *d_th, float *d_diffcoef1, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad);
__global__ void cuda_backward_p_vti(float *d_p2, float *d_p1, float *d_p0, 
				    float *d_vp, float *d_ep, float *d_de, float *d_diffcoef1, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad);
__global__ void cuda_backward_p_iso(float *d_p2, float *d_p1, float *d_p0, 
				    float *d_vp, float *d_diffcoef2, int *d_norderx, int *d_norderz, 
				    float dt, float dx, float dz, int npml, int nxpad, int nzpad);
//==============================================================================
__global__ void save_pmllr_f(float *d_p, float *d_pmlplr, int nxpad, int nzpad, int npml, int Ntemp);
__global__ void save_pmltb_f(float *d_p, float *d_pmlptb, int nxpad, int nzpad, int npml, int Ntemp);
__global__ void read_pmllr_f(float *d_p, float *d_pmlplr, int nxpad, int nzpad, int npml, int Ntemp);
__global__ void read_pmltb_f(float *d_p, float *d_pmlptb, int nxpad, int nzpad, int npml, int Ntemp);
//==============================================================================
__global__ void cuda_imagingconditon_lsrtm(float *d_ps2, float *d_ps1, float *d_ps0, float *d_pr1,float *d_image, float *d_illum, float dt, int nxpad, int nzpad, int npml);

void check_gpu_error(const char *msg);

#endif
