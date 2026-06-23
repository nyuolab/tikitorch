// forge_pc_train_f2.c — TRAIN a net ENTIRELY ON THE FPGA. The host only stages a batch and kicks the
// forge_pc_train_step kernel (one kick = forward + cascade backward + SignSGD-momentum update); weights +
// momentum persist in on-card DDR across kicks. Periodically the host reads the weights back and evaluates
// test accuracy (a forward pass) to show it climbing -> training happens on multiplier-free silicon.
//   gcc -O2 -I$SDK/include forge_pc_train_f2.c -lfpga_mgmt -o forge_pc_train_f2 ; sudo ./forge_pc_train_f2 <slot> <bindir> <epochs>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define D 256
#define N 32
#define C 10
#define H1_BITS 7
#define LG_BITS 4
#define O_W0   0
#define O_W1   (O_W0   + D*D)
#define O_MOM0 (O_W1   + D*D)
#define O_MOM1 (O_MOM0 + D*D)
#define O_X    (O_MOM1 + D*D)
#define O_TGT  (O_X    + N*D)
#define BUFLEN (O_TGT  + N*D)
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rshift(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=h-t; return s>0?s:0; }
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static int8_t *load(const char*p,long*n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);} fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET); int8_t*b=malloc(*n); if(fread(b,1,*n,f)!=(size_t)*n){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return b; }

static pci_bar_handle_t bar0, bar4, bar4r;
static int32_t *hbuf;

static void stage(int off, int n){ for(long o=off; o<off+n; o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4, (uint64_t)o*4, (uint32_t*)(hbuf+o), (uint64_t)k);} }
static void kick(){ fpga_pci_poke(bar0,R_BUF0,0); fpga_pci_poke(bar0,R_BUF1,0); fpga_pci_poke(bar0,R_AP_CTRL,AP_START);
    double t0=now_s(); uint32_t c=0; while(now_s()-t0<5.0){ fpga_pci_peek(bar0,R_AP_CTRL,&c); if(c&AP_DONE)return;} fprintf(stderr,"KICK HUNG\n"); exit(6); }
static void readw(int off, int32_t*dst, int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r,(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }

// host forward (eval): == the kernel forward (dyad matmul + relu + max-shift), returns predicted class
static int predict(const int8_t*x, const int32_t*W0, const int32_t*W1){
    long pre0[D]; long mxh=0;
    for(int o=0;o<D;o++){ long a=0; for(int i=0;i<D;i++) a+=rdyad(W0[o*D+i])*x[i]; pre0[o]=a; long r=a>0?a:0; if(r>mxh)mxh=r; }
    int sh0=rshift(mxh,H1_BITS); long h1[D];
    for(int o=0;o<D;o++){ long r=pre0[o]>0?pre0[o]:0; h1[o]=(r>>sh0); if(h1[o]>32767)h1[o]=32767; }
    long lg[C]; long mxl=0;
    for(int c=0;c<C;c++){ long a=0; for(int i=0;i<D;i++) a+=rdyad(W1[c*D+i])*h1[i]; lg[c]=a; long al=a<0?-a:a; if(al>mxl)mxl=al; }
    int sh1=rshift(mxl,LG_BITS); int best=0; long bv=-1L<<62;
    for(int c=0;c<C;c++){ long v=lg[c]>>sh1; if(v>bv){bv=v;best=c;} } return best;
}

int main(int argc,char**argv){
    int slot=argc>1?atoi(argv[1]):0; const char*dir=argc>2?argv[2]:"/tmp/pc_mnist"; int epochs=argc>3?atoi(argv[3]):3;
    long ntr,nte,nytr,nyte; char p[512];
    snprintf(p,512,"%s/train_x.i8",dir); int8_t*Xtr=load(p,&ntr);
    snprintf(p,512,"%s/train_y.u8",dir); int8_t*Ytr=load(p,&nytr);
    snprintf(p,512,"%s/test_x.i8",dir);  int8_t*Xte=load(p,&nte);
    snprintf(p,512,"%s/test_y.u8",dir);  int8_t*Yte=load(p,&nyte);
    int Ntr=nytr, Nte=nyte;
    printf("[train] data: %d train, %d test (D=%d)\n", Ntr, Nte, D);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR0,0,&bar0)){fprintf(stderr,"bar0\n");return 3;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4)){fprintf(stderr,"bar4\n");return 4;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,0,&bar4r)){fprintf(stderr,"bar4r\n");return 4;}
    hbuf=calloc(BUFLEN,4);

    // init weights (small random int), zero momentum; stage all once -> persist in DDR
    srand(12345);
    for(int i=0;i<D*D;i++){ hbuf[O_W0+i]=(rand()%9)-4; hbuf[O_W1+i]=(rand()%9)-4; hbuf[O_MOM0+i]=0; hbuf[O_MOM1+i]=0; }
    int32_t* initW0=malloc(4*D*D); memcpy(initW0, hbuf+O_W0, 4*D*D);   // snapshot init to detect evolution
    stage(0, BUFLEN);
    // diag: 1 kick, then read W0/W1 back -> did the kernel's update persist in DDR across the kick?
    { for(int n=0;n<N;n++){ for(int i=0;i<D;i++) hbuf[O_X+n*D+i]=Xtr[n*D+i]; for(int c=0;c<C;c++) hbuf[O_TGT+n*D+c]=0; hbuf[O_TGT+n*D+Ytr[n]]=8; }
      stage(O_X, 2*N*D); kick();
      int32_t* w=malloc(4*D*D); readw(O_W0,w,D*D); long ch=0; for(int j=0;j<D*D;j++) if(w[j]!=initW0[j]) ch++;
      printf("[diag] after 1 kick: %ld/%d W0 weights changed from init (W0[0]:%d->%d, W0[1]:%d->%d)\n",
             ch, D*D, initW0[0], w[0], initW0[1], w[1]);
      free(w); fflush(stdout); }

    int32_t *W0=malloc(4*D*D), *W1=malloc(4*D*D);
    int nb = Ntr / N;
    for(int ep=0; ep<epochs; ep++){
        double t0=now_s();
        for(int b=0; b<nb; b++){
            for(int n=0;n<N;n++){ int s=b*N+n;
                for(int i=0;i<D;i++) hbuf[O_X+n*D+i]=Xtr[s*D+i];
                for(int c=0;c<C;c++) hbuf[O_TGT+n*D+c]=0;
                hbuf[O_TGT+n*D+Ytr[s]]=8;
            }
            stage(O_X, 2*N*D);          // stage just this batch's x+tgt (weights/mom persist in DDR)
            kick();                      // one on-fabric training step
        }
        readw(O_W0,W0,D*D); readw(O_W1,W1,D*D);   // read trained weights back, eval on host
        long ok=0; for(int t=0;t<Nte;t++) if(predict(Xte+t*D,W0,W1)==Yte[t]) ok++;
        printf("[train] epoch %d: test acc = %.4f  (%d kicks, %.1fs)  ON-FABRIC\n", ep+1, (double)ok/Nte, nb, now_s()-t0);
        fflush(stdout);
    }
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}
