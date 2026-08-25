#include <stdint.h>
#include <stdio.h>

enum { LEAF, MATMUL, RELU, MSE, BIAS };
enum { ROLE_INPUT, ROLE_PARAMETER, ROLE_CONSTANT, ROLE_TEMPORARY };
enum { NEED_LHS=1, NEED_RHS=2 };
typedef struct { uint8_t op,role,trainable,pad; uint32_t lhs,rhs; } Node;

static unsigned compile_masks(const Node *nodes,unsigned count,uint8_t *depends,uint8_t *masks){
    unsigned edges=0;
    for(unsigned i=0;i<count;i++){
        if(nodes[i].op==LEAF){depends[i]=(nodes[i].role==ROLE_PARAMETER&&nodes[i].trainable);continue;}
        uint8_t mask=0;
        if(depends[nodes[i].lhs])mask|=NEED_LHS;
        if(nodes[i].op!=RELU&&depends[nodes[i].rhs])mask|=NEED_RHS;
        masks[i]=mask;depends[i]=mask!=0;
        edges+=(mask&NEED_LHS)!=0;edges+=(mask&NEED_RHS)!=0;
    }
    return edges;
}

int main(void){
    Node n[12]={
      {LEAF,ROLE_INPUT,0,0,0,0},
      {LEAF,ROLE_PARAMETER,1,0,0,0},
      {LEAF,ROLE_PARAMETER,1,0,0,0},
      {LEAF,ROLE_PARAMETER,0,0,0,0},
      {LEAF,ROLE_PARAMETER,0,0,0,0},
      {LEAF,ROLE_CONSTANT,0,0,0,0},
      {MATMUL,ROLE_TEMPORARY,0,0,0,1},
      {BIAS,ROLE_TEMPORARY,0,0,6,2},
      {RELU,ROLE_TEMPORARY,0,0,7,0},
      {MATMUL,ROLE_TEMPORARY,0,0,8,3},
      {BIAS,ROLE_TEMPORARY,0,0,9,4},
      {MSE,ROLE_TEMPORARY,0,0,10,5}
    };
    uint8_t depends[12]={0},masks[12]={0};
    unsigned edges=compile_masks(n,12,depends,masks);
    if(masks[6]!=NEED_RHS || masks[7]!=(NEED_LHS|NEED_RHS) || masks[8]!=NEED_LHS ||
       masks[9]!=NEED_LHS || masks[10]!=NEED_LHS || masks[11]!=NEED_LHS || edges!=7)return 1;
    n[1].trainable=n[2].trainable=0;
    for(unsigned i=0;i<12;i++)depends[i]=masks[i]=0;
    if(compile_masks(n,12,depends,masks)!=0 || depends[11])return 1;
    puts("needs-grad spike passed: edge masks prune frozen/input/constant gradients; all-frozen backward=empty");
    return 0;
}
