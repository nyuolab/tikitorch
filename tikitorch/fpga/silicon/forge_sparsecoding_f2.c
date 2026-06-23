// forge_sparsecoding_f2.c — train a GENERATIVE sparse-coding dictionary ENTIRELY on the FPGA, then RECOVER
// masked MNIST by settling on the FPGA. The host stages a batch + kicks forge_sparsecoding_pc; the dyadic
// dictionary persists in on-card DDR across kicks (mode=0 train). Then one mode=1 kick settles the code for
// 50%-masked test images and the host reads back the reconstruction -> in-paints. ONE trained model answers
// the in-paint query by settling, multiplier-free, no backprop. Dumps orig/mask/infill (.i32) for rendering.
//   gcc -O2 -I$SDK/include forge_sparsecoding_f2.c -lfpga_mgmt -o forge_sparsecoding_f2
//   sudo ./forge_sparsecoding_f2 <slot> <bindir> <epochs>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define P 256
#define M 256
#define NB 32
#define PIX 64          // pixel scale (bins are 0..16 -> x4 -> 0..64; matches the fabric-ref)
#define WL 16
#define O_CTRL  0
#define O_D     (O_CTRL  + WL)
#define O_X     (O_D     + P*M)
#define O_MASK  (O_X     + NB*P)
#define O_S     (O_MASK  + NB*P)
#define O_RECON (O_S     + NB*M)
#define BUFLEN  (O_RECON + NB*P)
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int8_t *load(const char*p,long*n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);} fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET); int8_t*b=malloc(*n); if(fread(b,1,*n,f)!=(size_t)*n){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return b; }
static pci_bar_handle_t bar0,bar4,bar4r; static int32_t *hbuf;
static void stage(int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4,(uint64_t)o*4,(uint32_t*)(hbuf+o),(uint64_t)k);} }
static void kick(){ fpga_pci_poke(bar0,R_BUF0,0); fpga_pci_poke(bar0,R_BUF1,0); fpga_pci_poke(bar0,R_AP_CTRL,AP_START);
    double t0=now_s(); uint32_t c=0; while(now_s()-t0<10.0){ fpga_pci_peek(bar0,R_AP_CTRL,&c); if(c&AP_DONE)return; } fprintf(stderr,"KICK HUNG\n"); exit(6); }
static void readw(int off,int32_t*dst,int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r,(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }
static void setctrl(int mode,int nsteps){ hbuf[O_CTRL+0]=mode; hbuf[O_CTRL+1]=nsteps; stage(O_CTRL,WL); }

int main(int argc,char**argv){
    int slot=argc>1?atoi(argv[1]):0; const char*dir=argc>2?argv[2]:"/tmp/pc_mnist"; int epochs=argc>3?atoi(argv[3]):12;
    long ntr,nte,nytr,nyte; char p[512];
    snprintf(p,512,"%s/train_x.i8",dir); int8_t*Xtr=load(p,&ntr);
    snprintf(p,512,"%s/train_y.u8",dir); (void)load((snprintf(p,512,"%s/train_y.u8",dir),p),&nytr);
    snprintf(p,512,"%s/test_x.i8",dir);  int8_t*Xte=load(p,&nte);
    snprintf(p,512,"%s/test_y.u8",dir);  (void)load((snprintf(p,512,"%s/test_y.u8",dir),p),&nyte);
    int Ntr=ntr/P, Nte=nte/P;
    printf("[scpc] generative sparse-coding on F2: %d train %d test, P=%d M=%d\n", Ntr, Nte, P, M);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR0,0,&bar0)){fprintf(stderr,"bar0\n");return 3;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4)){fprintf(stderr,"bar4\n");return 4;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,0,&bar4r)){fprintf(stderr,"bar4r\n");return 4;}
    hbuf=calloc(BUFLEN,4);

    srand(12345);
    for(int i=0;i<P*M;i++){ double g=0; for(int k=0;k<12;k++) g+=rand()/(double)RAND_MAX; hbuf[O_D+i]=(int)(4.0*(g-6.0)+(g>=6.0?0.5:-0.5)); }  // randn*4
    stage(0,BUFLEN);
    // ---- TRAIN the dictionary ON FABRIC (mode=0; D persists + updates in DDR across kicks) ----
    int nb=Ntr/NB;
    for(int ep=0; ep<epochs; ep++){
        double t0=now_s();
        for(int b=0;b<nb;b++){
            for(int n=0;n<NB;n++){ int s=b*NB+n; for(int i=0;i<P;i++) hbuf[O_X+n*P+i]=(int)Xtr[s*P+i]*PIX/16; for(int i=0;i<P;i++) hbuf[O_MASK+n*P+i]=1; }
            stage(O_X, 2*NB*P);          // x + mask
            setctrl(0,20); kick();        // on-fabric train kick
        }
        printf("[scpc] epoch %d done (%d kicks, %.1fs)  ON-FABRIC dict training\n", ep+1, nb, now_s()-t0); fflush(stdout);
    }
    // ---- IN-PAINT: settle 50%-masked test images ON FABRIC (mode=1), read back the reconstruction ----
    int32_t *orig=malloc(4*NB*P), *msk=malloc(4*NB*P), *infill=malloc(4*NB*P), *recon=malloc(4*NB*P);
    for(int n=0;n<NB;n++){ for(int i=0;i<P;i++){ int x=(int)Xte[n*P+i]*PIX/16; orig[n*P+i]=x; int mo=(rand()%2); msk[n*P+i]=mo;
        hbuf[O_X+n*P+i]=mo?x:0; hbuf[O_MASK+n*P+i]=mo; } }
    stage(O_X, 2*NB*P); setctrl(1,120); kick();
    readw(O_RECON, recon, NB*P);
    for(int i=0;i<NB*P;i++) infill[i] = msk[i]? orig[i] : recon[i];
    // dump orig/mask/infill for host rendering + report in-paint MSE vs mean-fill
    FILE*f;
    f=fopen("/tmp/sc_orig.i32","wb");   fwrite(orig,4,NB*P,f);   fclose(f);
    f=fopen("/tmp/sc_mask.i32","wb");   fwrite(msk,4,NB*P,f);    fclose(f);
    f=fopen("/tmp/sc_infill.i32","wb"); fwrite(infill,4,NB*P,f); fclose(f);
    double se=0, sm=0; long nm=0;
    for(int n=0;n<NB;n++){ long sum=0,cnt=0; for(int i=0;i<P;i++) if(msk[n*P+i]){sum+=orig[n*P+i];cnt++;} double mean=cnt?(double)sum/cnt:0;
        for(int i=0;i<P;i++) if(!msk[n*P+i]){ double d=infill[n*P+i]-orig[n*P+i]; se+=d*d; double dm=mean-orig[n*P+i]; sm+=dm*dm; nm++; } }
    printf("[scpc] 50%%-mask in-paint ON FABRIC: MSE=%.2f vs mean-fill %.2f  (%.1f%% better)  (%ld missing px)\n",
           se/nm, sm/nm, 100.0*(1.0-se/sm), nm); fflush(stdout);
    { int32_t D0[8]; readw(O_D,D0,8); printf("[scpc] dumped sc_{orig,mask,infill}.i32 (D[0..3]=%d %d %d %d)\n",D0[0],D0[1],D0[2],D0[3]); }
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}
