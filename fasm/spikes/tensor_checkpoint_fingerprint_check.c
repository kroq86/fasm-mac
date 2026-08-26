#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { INPUT, PARAM, MATMUL, BIAS, RELU, MSE };
typedef struct {
    uint32_t id, op, inputs[2], input_count, dtype, rank, dims[4], semantic_flags;
    char name[16];
} Node;
typedef struct { const Node *nodes; uint32_t count, output_id; } Graph;

static void hash_bytes(uint64_t *h, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    for (size_t i = 0; i < n; i++) { *h ^= p[i]; *h *= UINT64_C(1099511628211); }
}
static void hash32(uint64_t *h, uint32_t v) {
    unsigned char le[4] = {v, v >> 8, v >> 16, v >> 24}; hash_bytes(h, le, 4);
}
static void hash64(uint64_t *h, uint64_t v) {
    unsigned char le[8]; for(unsigned i=0;i<8;i++)le[i]=(unsigned char)(v>>(8*i)); hash_bytes(h,le,8);
}
/* Canonical semantic hash. Runtime backend, fusion, plan order, scratch
   offsets and executor ABI are deliberately absent. */
static const Node *find_node(Graph graph, uint32_t id) {
    for (uint32_t i=0;i<graph.count;i++) if(graph.nodes[i].id==id)return &graph.nodes[i];
    return NULL;
}
static uint64_t node_fingerprint(Graph graph,uint32_t id,uint32_t depth) {
    const Node*n=find_node(graph,id);if(!n||depth>graph.count)return 0;
    uint64_t h = UINT64_C(1469598103934665603);
    hash32(&h,n->op);hash32(&h,n->input_count);hash32(&h,n->dtype);hash32(&h,n->rank);
    for(uint32_t j=0;j<n->rank;j++)hash32(&h,n->dims[j]);hash32(&h,n->semantic_flags);
    if(n->op==PARAM)hash_bytes(&h,n->name,strnlen(n->name,sizeof n->name)+1);
    for(uint32_t j=0;j<n->input_count;j++){uint64_t child=node_fingerprint(graph,n->inputs[j],depth+1);if(!child)return 0;hash64(&h,child);}
    return h;
}
static uint64_t fingerprint(Graph graph) {
    uint64_t root=node_fingerprint(graph,graph.output_id,0),h=UINT64_C(1469598103934665603);if(!root)return 0;
    hash_bytes(&h,"FASMGRAPH1",10);hash64(&h,root);return h;
}

static Graph base_graph(Node *n) {
    n[0]=(Node){.id=10,.op=INPUT,.dtype=1,.rank=2,.dims={1,2}};
    n[1]=(Node){.id=20,.op=PARAM,.dtype=1,.rank=2,.dims={2,4},.name="w1"};
    n[2]=(Node){.id=30,.op=MATMUL,.inputs={10,20},.input_count=2,.dtype=1,.rank=2,.dims={1,4}};
    n[3]=(Node){.id=40,.op=RELU,.inputs={30},.input_count=1,.dtype=1,.rank=2,.dims={1,4}};
    return (Graph){n,4,40};
}

int main(void) {
    Node a[4], reordered[4], changed_op[4], changed_shape[4], changed_edge[4];
    Graph ga=base_graph(a); uint64_t expected=fingerprint(ga); if(!expected)return 1;
    reordered[0]=a[3];reordered[1]=a[1];reordered[2]=a[0];reordered[3]=a[2];
    if(fingerprint((Graph){reordered,4,40})!=expected)return 2;
    Node renumbered[4];memcpy(renumbered,a,sizeof a);uint32_t ids[]={7,99,12,3};
    for(unsigned i=0;i<4;i++)renumbered[i].id=ids[i];renumbered[2].inputs[0]=7;renumbered[2].inputs[1]=99;renumbered[3].inputs[0]=12;
    if(fingerprint((Graph){renumbered,4,3})!=expected)return 3;
    memcpy(changed_op,a,sizeof a);changed_op[3].op=BIAS;
    memcpy(changed_shape,a,sizeof a);changed_shape[1].dims[1]=5;
    memcpy(changed_edge,a,sizeof a);changed_edge[2].inputs[1]=10;
    if(fingerprint((Graph){changed_op,4,40})==expected||fingerprint((Graph){changed_shape,4,40})==expected||fingerprint((Graph){changed_edge,4,40})==expected)return 4;
    /* These execution choices are intentionally not arguments to fingerprint. */
    const char *backends[]={"scalar","neon","avx2"};uint32_t plan_steps[]={4,2,3};
    for(unsigned i=0;i<3;i++){(void)backends[i];(void)plan_steps[i];if(fingerprint(ga)!=expected)return 5;}
    printf("tensor checkpoint fingerprint spike passed: fingerprint=%016llx node_order/id=excluded semantic_op/shape/edge/parameter_name=bound backend/fusion/plan/scratch/executor_abi=excluded\n",(unsigned long long)expected);
    return 0;
}
