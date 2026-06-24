// forge_pc_train_shard_net.c — train ONE net MODEL-PARALLEL across TWO f2 INSTANCES (cross-box, 2*L_local
// physical FPGAs) with exactly ONE neighbour message crossing the VPC network per direction per step. This is
// the 2-FPGA / 8-FPGA distributed-PC rig (forge_pc_block AGFI, residual deep MLP) extended to span two
// physically separate accelerators: the head instance owns layers 0..L_local-1, the tail owns L_local..2L-1;
// the boundary activation (head's last layer -> tail's first, forward) and the boundary error (tail's first
// layer -> head's last, backward) cross by TCP. Everything else (intra-box neighbour messages, weights,
// gradients) stays local — Gate C extended with an xinstance_msg counter. STRUCTURAL demo: TCP < PCIe < fabric,
// so this is slower wall-clock than one box; the point is that the no-global-coordination scheme runs across
// two boxes with a single network link. Reuses forge_pc_train_shard.c's kernel arithmetic + residual skips.
//   gcc -O2 -std=gnu11 -I$SDK/include forge_pc_train_shard_net.c -lfpga_mgmt -o shardnet
//   HEAD: sudo ROLE=head PEER=<tailIP> PORT=5599 ./shardnet <epochs> <bindir> <slot0..slotL-1>
//   TAIL: sudo ROLE=tail LISTEN=1 PORT=5599 ./shardnet <epochs> <bindir> <slot0..slotL-1>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define D 256
#define N 32
#define C 10
#define H1_BITS 7
#define LG_BITS 4
#define EDN_BITS 13
#define MAXL 8
#define WL 16
#define O_CFG  0
#define O_W    (O_CFG  + WL)
#define O_MOM  (O_W    + D*D)
#define O_AIN  (O_MOM  + D*D)
#define O_TGT  (O_AIN  + N*D)
#define O_EIN  (O_TGT  + N*D)
#define O_AOUT (O_EIN  + N*D)
#define O_MASK (O_AOUT + N*D)
#define O_EOUT (O_MASK + N*D)
#define BUFLEN (O_EOUT + N*D)
#define PH_FWD 0
#define PH_BWD 1
#define RL_HIDDEN 0
#define RL_OUTPUT 1
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rshift(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=(h+1)-t; return s>0?s:0; }
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int8_t *load(const char*p,long*n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);} fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET); int8_t*b=malloc(*n); if(fread(b,1,*n,f)!=(size_t)*n){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return b; }

static int L;                                 // local layers on THIS instance
static int HEAD;                              // 1 = head (layers 0..L-1, owns input), 0 = tail (owns output)
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];
static int sock = -1;                         // the cross-instance TCP link

static void stage(int l,int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} }
static void kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    double t0=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); uint32_t c=0;
    while(now_s()-t0<8.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c); if(c&AP_DONE)return; } fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static void readw(int l,int off,int32_t*dst,int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }
static void setcfg(int l,int phase,int role,int doupd){ hbuf[l][O_CFG+0]=phase; hbuf[l][O_CFG+1]=role; hbuf[l][O_CFG+2]=8; hbuf[l][O_CFG+3]=doupd; stage(l,O_CFG,WL); }

// blocking send/recv of an N*D int32 vector over the cross-instance link
static long xinstance_msg = 0;
static void net_send(const int32_t*v,int n){ size_t b=(size_t)n*4, off=0; const char*p=(const char*)v;
    while(off<b){ ssize_t k=send(sock,p+off,b-off,0); if(k<=0){perror("send");exit(7);} off+=k; } xinstance_msg+=n; }
static void net_recv(int32_t*v,int n){ size_t b=(size_t)n*4, off=0; char*p=(char*)v;
    while(off<b){ ssize_t k=recv(sock,p+off,b-off,0); if(k<=0){perror("recv");exit(7);} off+=k; } xinstance_msg+=n; }

static int role_of(int l){ return (!HEAD && l==L-1) ? RL_OUTPUT : RL_HIDDEN; }   // OUTPUT only = tail's last layer

int main(int argc,char**argv){
    int epochs = argc>1?atoi(argv[1]):4;
    const char*dir = argc>2?argv[2]:"/tmp/pc_mnist";
    L = argc-3; if(L<1||L>MAXL){ fprintf(stderr,"usage: %s <epochs> <bindir> <slot..> (1<=L<=%d)\n",argv[0],MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);
    HEAD = getenv("ROLE") && !strcmp(getenv("ROLE"),"head");
    int PORT = getenv("PORT")?atoi(getenv("PORT")):5599;
    const char* ipdir = getenv("PC_INITPERM");

    // ---- establish the cross-instance TCP link (tail listens, head connects) ----
    if(getenv("LISTEN")){
        int ls=socket(AF_INET,SOCK_STREAM,0); int one=1; setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
        struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(PORT);
        if(bind(ls,(struct sockaddr*)&a,sizeof a)){perror("bind");return 3;} listen(ls,1);
        fprintf(stderr,"[net] listening on :%d ...\n",PORT); sock=accept(ls,0,0); if(sock<0){perror("accept");return 3;}
    } else {
        const char* peer=getenv("PEER"); if(!peer){fprintf(stderr,"set PEER=<ip>\n");return 1;}
        sock=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(PORT);
        inet_pton(AF_INET,peer,&a.sin_addr);
        for(int tries=0;;tries++){ if(connect(sock,(struct sockaddr*)&a,sizeof a)==0)break; if(tries>120){perror("connect");return 3;} usleep(500000); sock=socket(AF_INET,SOCK_STREAM,0); a.sin_family=AF_INET; a.sin_port=htons(PORT); inet_pton(AF_INET,peer,&a.sin_addr); }
    }
    { int one=1; setsockopt(sock,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one); }
    fprintf(stderr,"[net] cross-instance link up (%s)\n", HEAD?"head":"tail");

    long ntr,nte,nytr,nyte; char p[512];
    snprintf(p,512,"%s/train_x.i8",dir); int8_t*Xtr=load(p,&ntr);
    snprintf(p,512,"%s/train_y.u8",dir); int8_t*Ytr=load(p,&nytr);
    snprintf(p,512,"%s/test_x.i8",dir);  int8_t*Xte=load(p,&nte);
    snprintf(p,512,"%s/test_y.u8",dir);  int8_t*Yte=load(p,&nyte);
    int Ntr=nytr, Nte=nyte; (void)Xte;(void)Yte;(void)Nte;

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r %d\n",slot[l]);return 4;}
        hbuf[l]=calloc(BUFLEN,4);
    }
    // init weights: load the verified per-GLOBAL-layer init (head: W0..; tail: W{L}..W{2L-1}) so the two boxes
    // form ONE net. PC_INITPERM holds W0_init..W{2L-1}_init + perm; this instance loads its slice.
    int goff = HEAD ? 0 : L;                          // global-layer offset of this instance's first layer
    srand(12345);
    for(int l=0;l<L;l++){
        if(ipdir){ snprintf(p,512,"%s/W%d_init.i32",ipdir,goff+l); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
                   if(fread(hbuf[l]+O_W,4,D*D,f)!=(size_t)(D*D)){fprintf(stderr,"read %s\n",p);return 2;} fclose(f); }
        else for(int i=0;i<D*D;i++){ double g=0; for(int k=0;k<12;k++) g+=rand()/(double)RAND_MAX; hbuf[l][O_W+i]=(int)(4.0*(g-6.0)+(g>=6.0?0.5:-0.5)); }
        for(int i=0;i<D*D;i++) hbuf[l][O_MOM+i]=0;
        stage(l,0,BUFLEN);
    }
    int *perm=malloc(Ntr*sizeof(int)); for(int i=0;i<Ntr;i++) perm[i]=i;
    if(ipdir){ snprintf(p,512,"%s/perm.i32",ipdir); FILE*f=fopen(p,"rb"); if(f){ if(fread(perm,4,Ntr,f)!=(size_t)Ntr){fprintf(stderr,"read perm\n");} fclose(f);} }
    int32_t *aout=malloc(4*N*D), *eout=malloc(4*N*D), *bmsg=malloc(4*N*D);
    int RES=1;                                       // residual skips on (deep cross-box net)
    int nb=Ntr/N; long xchip_msg=0; double t0=now_s();
    fprintf(stderr,"[net] %s: %d local layers (global %d..%d), 2-instance run, %d steps/epoch\n",
            HEAD?"HEAD":"TAIL", L, goff, goff+L-1, nb);

    for(int ep=0; ep<epochs; ep++){
        for(int b=0;b<nb;b++){
            // ================= FORWARD =================
            if(HEAD){
                int32_t curin[N*D];
                for(int l=0;l<L;l++){
                    setcfg(l,PH_FWD,RL_HIDDEN,0);
                    if(l==0){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int i=0;i<D;i++) hbuf[0][O_AIN+n*D+i]=Xtr[s*D+i]; } }
                    else    { memcpy(hbuf[l]+O_AIN, aout, 4*N*D); xchip_msg+=(long)N*D; }
                    stage(l,O_AIN,N*D);
                    kick(l);
                    readw(l,O_AOUT,aout,N*D);
                    if(RES) for(int j=0;j<N*D;j++) aout[j]+=hbuf[l][O_AIN+j];   // residual skip (all head layers hidden)
                }
                net_send(aout,N*D);                  // boundary activation -> tail (the ONLY fwd cross-box msg)
            } else {
                net_recv(bmsg,N*D);                  // boundary activation from head
                for(int l=0;l<L;l++){
                    setcfg(l,PH_FWD,role_of(l),0);
                    if(l==0) memcpy(hbuf[0]+O_AIN, bmsg, 4*N*D);
                    else   { memcpy(hbuf[l]+O_AIN, aout, 4*N*D); xchip_msg+=(long)N*D; }
                    stage(l,O_AIN,N*D);
                    kick(l);
                    if(l<L-1){ readw(l,O_AOUT,aout,N*D); if(RES) for(int j=0;j<N*D;j++) aout[j]+=hbuf[l][O_AIN+j]; }
                }
            }
            // ================= BACKWARD =================
            if(!HEAD){
                for(int l=L-1;l>=0;l--){
                    setcfg(l,PH_BWD,role_of(l),1);
                    if(l==L-1){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int c=0;c<D;c++) hbuf[l][O_TGT+n*D+c]=0; hbuf[l][O_TGT+n*D+Ytr[s]]=8; } stage(l,O_TGT,N*D); }
                    else      { memcpy(hbuf[l]+O_EIN, eout, 4*N*D); xchip_msg+=(long)N*D; stage(l,O_EIN,N*D); }
                    kick(l);
                    if(l>0){ readw(l,O_EOUT,eout,N*D);
                        if(RES && l<L-1){ for(int j=0;j<N*D;j++) eout[j]+=hbuf[l][O_EIN+j];
                            long mx=0; for(int j=0;j<N*D;j++){ long a=eout[j]<0?-eout[j]:eout[j]; if(a>mx)mx=a; } int sh=rshift(mx,EDN_BITS); for(int j=0;j<N*D;j++) eout[j]>>=sh; } }
                }
                // tail's layer-0 EOUT is the boundary error -> head (the ONLY bwd cross-box msg).
                readw(0,O_EOUT,eout,N*D); net_send(eout,N*D);
            } else {
                net_recv(bmsg,N*D);                  // boundary error from tail = grad wrt head's last output
                for(int l=L-1;l>=0;l--){
                    setcfg(l,PH_BWD,RL_HIDDEN,1);
                    if(l==L-1) memcpy(hbuf[l]+O_EIN, bmsg, 4*N*D);
                    else     { memcpy(hbuf[l]+O_EIN, eout, 4*N*D); xchip_msg+=(long)N*D; }
                    // residual skip across the boundary uses this layer's incoming error (EIN) just staged
                    stage(l,O_EIN,N*D);
                    kick(l);
                    if(l>0){ readw(l,O_EOUT,eout,N*D);
                        for(int j=0;j<N*D;j++) eout[j]+=hbuf[l][O_EIN+j];      // residual (every head layer hidden)
                        long mx=0; for(int j=0;j<N*D;j++){ long a=eout[j]<0?-eout[j]:eout[j]; if(a>mx)mx=a; } int sh=rshift(mx,EDN_BITS); for(int j=0;j<N*D;j++) eout[j]>>=sh; }
                }
            }
        }
        fprintf(stderr,"[net] %s epoch %d done (%.1fs)\n", HEAD?"HEAD":"TAIL", ep+1, now_s()-t0);
    }
    double wall=now_s()-t0; long steps=(long)epochs*nb;
    fprintf(stderr,"[net] %s DONE: %ld steps, %.1fs, %.1f steps/s | intra-box xchip=%ld ints, CROSS-INSTANCE xinstance=%ld ints (%.2f MB)\n",
            HEAD?"HEAD":"TAIL", steps, wall, steps/wall, xchip_msg, xinstance_msg, xinstance_msg*4/1e6);
    fprintf(stderr,"[gateC] WEIGHTS crossed instances=0 | GRADIENTS crossed=0 | only the boundary activation+error crossed the network\n");
    for(int l=0;l<L;l++){ snprintf(p,512,"/tmp/netW%d.i32",goff+l); FILE*f=fopen(p,"wb"); if(f){ int32_t*W=malloc(4*D*D); readw(l,O_W,W,D*D); fwrite(W,4,D*D,f); fclose(f); free(W);} }
    close(sock);
    for(int l=0;l<L;l++){ fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
