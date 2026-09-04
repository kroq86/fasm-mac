/* Instrument-sensitivity spike for a verified state-transition agent. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { int battery, temperature, critical_power, generator_ok; uint64_t revision; } World;
typedef enum { ACT_CHARGE=1, ACT_COOL=2, ACT_SHED=3 } Action;
typedef struct { Action action; int amount; uint64_t expected_revision; } Proposal;
typedef struct { World before, after; Proposal proposal; int verdict; } Capsule;
enum { CAPSULE_BYTES = 92 };
static const uint64_t EXECUTOR_FINGERPRINT = UINT64_C(0x74656e736f723031);

static void put32(uint8_t **p, uint32_t v) { for(int i=0;i<4;i++) *(*p)++=(uint8_t)(v>>(8*i)); }
static void put64(uint8_t **p, uint64_t v) { for(int i=0;i<8;i++) *(*p)++=(uint8_t)(v>>(8*i)); }
static uint32_t get32(const uint8_t **p) { uint32_t v=0; for(int i=0;i<4;i++) v|=(uint32_t)*(*p)++<<(8*i); return v; }
static uint64_t get64(const uint8_t **p) { uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)*(*p)++<<(8*i); return v; }
static uint64_t checksum(const uint8_t *p, size_t n) {
    uint64_t h=UINT64_C(1469598103934665603);
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);} return h;
}
static void put_world(uint8_t **p,const World*w){put32(p,(uint32_t)w->battery);put32(p,(uint32_t)w->temperature);put32(p,(uint32_t)w->critical_power);put32(p,(uint32_t)w->generator_ok);put64(p,w->revision);}
static World get_world(const uint8_t **p){World w={(int32_t)get32(p),(int32_t)get32(p),(int32_t)get32(p),(int32_t)get32(p),get64(p)};return w;}
static int world_equal(const World*a,const World*b){return a->battery==b->battery&&a->temperature==b->temperature&&a->critical_power==b->critical_power&&a->generator_ok==b->generator_ok&&a->revision==b->revision;}

static int capsule_encode(const Capsule*c,uint8_t out[CAPSULE_BYTES]){
    if(!c||!out)return-1;uint8_t*p=out;
    put32(&p,UINT32_C(0x31434154));put32(&p,1);put64(&p,EXECUTOR_FINGERPRINT);
    put_world(&p,&c->before);put32(&p,(uint32_t)c->proposal.action);put32(&p,(uint32_t)c->proposal.amount);put64(&p,c->proposal.expected_revision);put_world(&p,&c->after);put32(&p,(uint32_t)c->verdict);
    put64(&p,checksum(out,CAPSULE_BYTES-8));return p==out+CAPSULE_BYTES?0:-2;
}

static int invariant(const World *w) {
    return w->battery >= 0 && w->battery <= 100 && w->temperature <= 95 &&
           (!w->critical_power || w->battery > 0 || w->generator_ok);
}

static int transition(World *w, const Proposal *p, Capsule *capsule, int verify) {
    World staged = *w;
    if (!p || !capsule || p->expected_revision != w->revision || p->amount <= 0) return -1;
    switch (p->action) {
    case ACT_CHARGE: staged.battery += p->amount; staged.temperature += p->amount / 2; break;
    case ACT_COOL: staged.temperature -= p->amount; break;
    case ACT_SHED: staged.critical_power = 0; break;
    default: return -2;
    }
    staged.revision++;
    *capsule = (Capsule){.before=*w,.after=staged,.proposal=*p,.verdict=0};
    if (verify && !invariant(&staged)) { capsule->verdict=-3; return -3; }
    *w = staged;
    return 0;
}

static int replay(const Capsule *c, World *out) {
    *out = c->before; Capsule replayed;
    int rc = transition(out, &c->proposal, &replayed, 1);
    return rc || !world_equal(out, &c->after) ? -1 : 0;
}

static int capsule_replay(const uint8_t bytes[CAPSULE_BYTES],World*out){
    if(!bytes||!out)return-10;const uint8_t*q=bytes+CAPSULE_BYTES-8;uint64_t expected=get64(&q);
    if(checksum(bytes,CAPSULE_BYTES-8)!=expected)return-10;
    const uint8_t*p=bytes;if(get32(&p)!=UINT32_C(0x31434154)||get32(&p)!=1||get64(&p)!=EXECUTOR_FINGERPRINT)return-11;
    Capsule c={0};c.before=get_world(&p);c.proposal.action=(Action)(int32_t)get32(&p);c.proposal.amount=(int32_t)get32(&p);c.proposal.expected_revision=get64(&p);c.after=get_world(&p);c.verdict=(int32_t)get32(&p);
    if(c.verdict)return-12;*out=c.before;Capsule actual;int rc=transition(out,&c.proposal,&actual,1);
    return rc||!world_equal(out,&c.after)?-13:0;
}

int main(void) {
    World initial={40,70,1,1,7}, a=initial; Capsule good;
    Proposal cool={ACT_COOL,10,7};
    if (transition(&a,&cool,&good,1) || a.temperature!=60 || a.revision!=8) return 1;
    World replayed; if (replay(&good,&replayed) || !world_equal(&a,&replayed)) return 2;
    uint8_t wire[CAPSULE_BYTES];if(capsule_encode(&good,wire)||capsule_replay(wire,&replayed)||!world_equal(&a,&replayed))return 6;
    uint8_t corrupt[CAPSULE_BYTES];memcpy(corrupt,wire,sizeof corrupt);corrupt[24]^=1;if(capsule_replay(corrupt,&replayed)!=-10)return 7;
    memcpy(corrupt,wire,sizeof corrupt);corrupt[4]=2;{uint8_t*p=corrupt+CAPSULE_BYTES-8;put64(&p,checksum(corrupt,CAPSULE_BYTES-8));}if(capsule_replay(corrupt,&replayed)!=-11)return 8;

    World b=initial, before=b; Capsule rejected;
    Proposal unsafe={ACT_CHARGE,70,7}; /* battery=110 and temperature=105 */
    if (transition(&b,&unsafe,&rejected,1)!=-3 || !world_equal(&b,&before)) return 3;

    /* Mutation control: disabling verification must create an observable
       invariant breach, proving that the endpoint can detect the mechanism. */
    World mutant=initial; Capsule mutation;
    if (transition(&mutant,&unsafe,&mutation,0) || invariant(&mutant)) return 4;

    Proposal stale={ACT_COOL,5,6};
    if (transition(&b,&stale,&rejected,1)!=-1 || !world_equal(&b,&before)) return 5;
    printf("agent transaction instrument passed: commit=atomic rollback=late-failure replay=canonical-bytes capsule-versioned=yes corruption=rejected stale-revision=rejected mutation-control=sensitive revision=%llu\n",
           (unsigned long long)a.revision);
    return 0;
}
