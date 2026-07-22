// TaskAlignedAssigner as plain C++ (no autograd — it's a non-differentiable match).
// Faithful port of v8loss.h assign(): candidates-in-gts, align metric, topk,
// multi-gt resolution, target gather, alignment normalization.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>

struct TALOut {
  std::vector<float> tb;   // (B*A*4) target boxes, image units
  std::vector<float> ts;   // (B*A*nc) target scores (normalized)
  std::vector<float> fg;   // (B*A) 0/1
};

static inline float tal_ciou(float x1,float y1,float x2,float y2,
                             float X1,float Y1,float X2,float Y2) {
  const float eps=1e-7f;
  float w1=x2-x1,h1=y2-y1,w2=X2-X1,h2=Y2-Y1;
  float iw=std::max(0.f,std::min(x2,X2)-std::max(x1,X1));
  float ih=std::max(0.f,std::min(y2,Y2)-std::max(y1,Y1));
  float inter=iw*ih, uni=w1*h1+w2*h2-inter+eps, iou=inter/uni;
  float cw=std::max(x2,X2)-std::min(x1,X1), ch=std::max(y2,Y2)-std::min(y1,Y1);
  float c2=cw*cw+ch*ch+eps;
  float rho2=((x1+x2-X1-X2)*(x1+x2-X1-X2)+(y1+y2-Y1-Y2)*(y1+y2-Y1-Y2))/4.f;
  float pi=(float)std::acos(-1.0);
  float a2=std::atan(w2/(h2+eps))-std::atan(w1/(h1+eps));
  float v=(4.f/(pi*pi))*a2*a2;
  float alpha=v/(v-iou+(1.f+eps));
  return iou-(rho2/c2+v*alpha);
}

// pd_scores (B,A,nc), pd_bboxes (B,A,4) image units, anc (A,2) image units,
// gt_labels (B,M), gt_bboxes (B,M,4), mask_gt (B,M).
inline TALOut tal_assign(const std::vector<float>& pd_scores, const std::vector<float>& pd_bboxes,
                         const std::vector<float>& anc, const std::vector<int64_t>& gt_labels,
                         const std::vector<float>& gt_bboxes, const std::vector<float>& mask_gt,
                         int64_t B,int64_t A,int64_t M,int64_t nc,int64_t topk,
                         float alpha,float beta) {
  const float EPS=1e-9f;
  TALOut o; o.tb.assign(B*A*4,0.f); o.ts.assign(B*A*nc,0.f); o.fg.assign(B*A,0.f);

  for (int64_t b=0;b<B;++b) {
    std::vector<float> align(M*A,0.f), overlaps(M*A,0.f);
    std::vector<char> maskin(M*A,0);
    for (int64_t m=0;m<M;++m) {
      float gx1=gt_bboxes[((b*M+m)*4)+0],gy1=gt_bboxes[((b*M+m)*4)+1];
      float gx2=gt_bboxes[((b*M+m)*4)+2],gy2=gt_bboxes[((b*M+m)*4)+3];
      int64_t lbl=gt_labels[b*M+m]; float mg=mask_gt[b*M+m];
      for (int64_t a=0;a<A;++a) {
        float ax=anc[a*2+0],ay=anc[a*2+1];
        float dmin=std::min(std::min(ax-gx1,ay-gy1),std::min(gx2-ax,gy2-ay));
        char in=(dmin>EPS)?1:0; maskin[m*A+a]=in;
        float mc=(in && mg>0)?1.f:0.f;
        float px1=pd_bboxes[((b*A+a)*4)+0],py1=pd_bboxes[((b*A+a)*4)+1];
        float px2=pd_bboxes[((b*A+a)*4)+2],py2=pd_bboxes[((b*A+a)*4)+3];
        float ov=std::max(0.f,tal_ciou(gx1,gy1,gx2,gy2,px1,py1,px2,py2))*mc;
        float sc=pd_scores[((b*A+a)*nc)+lbl]*mc;
        overlaps[m*A+a]=ov;
        align[m*A+a]=std::pow(sc,alpha)*std::pow(ov,beta);
      }
    }
    // topk per gt -> mask_pos
    std::vector<float> mask_pos(M*A,0.f);
    for (int64_t m=0;m<M;++m) {
      std::vector<int64_t> idx(A); std::iota(idx.begin(),idx.end(),0);
      std::partial_sort(idx.begin(),idx.begin()+topk,idx.end(),
                        [&](int64_t i,int64_t j){return align[m*A+i]>align[m*A+j];});
      for (int64_t t=0;t<topk;++t){ int64_t a=idx[t];
        if (maskin[m*A+a] && mask_gt[b*M+m]>0) mask_pos[m*A+a]=1.f; }
    }
    // fg + resolve multi-gt anchors by highest overlap
    for (int64_t a=0;a<A;++a){
      float s=0; for(int64_t m=0;m<M;++m) s+=mask_pos[m*A+a];
      if (s>1.f){ int64_t best=0; float bo=-1;
        for(int64_t m=0;m<M;++m){ if(overlaps[m*A+a]>bo){bo=overlaps[m*A+a];best=m;} }
        for(int64_t m=0;m<M;++m) mask_pos[m*A+a]=(m==best)?1.f:0.f; }
    }
    // alignment normalization factors
    std::vector<float> pos_am(M,0.f),pos_ov(M,0.f);
    for(int64_t m=0;m<M;++m){ float am=0,ov=0;
      for(int64_t a=0;a<A;++a){ float ap=align[m*A+a]*mask_pos[m*A+a];
        am=std::max(am,ap); ov=std::max(ov,overlaps[m*A+a]*mask_pos[m*A+a]); }
      pos_am[m]=am; pos_ov[m]=ov; }
    // targets
    for (int64_t a=0;a<A;++a){
      int64_t gt=-1; float best=0;
      for(int64_t m=0;m<M;++m){ if(mask_pos[m*A+a]>best){best=mask_pos[m*A+a];gt=m;} }
      float fg = 0; for(int64_t m=0;m<M;++m) fg+=mask_pos[m*A+a];
      o.fg[b*A+a]=(fg>0)?1.f:0.f;
      if (gt<0){ gt=0; }               // argmax default 0 when none (fg guards use)
      for(int j=0;j<4;++j) o.tb[((b*A+a)*4)+j]=gt_bboxes[((b*M+gt)*4)+j];
      if (fg>0){
        int64_t lbl=std::max<int64_t>(0,gt_labels[b*M+gt]);
        float norm=0;
        for(int64_t m=0;m<M;++m){ float ap=align[m*A+a]*mask_pos[m*A+a];
          float val=ap*pos_ov[m]/(pos_am[m]+EPS); norm=std::max(norm,val); }
        o.ts[((b*A+a)*nc)+lbl]=norm;
      }
    }
  }
  return o;
}
