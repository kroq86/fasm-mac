#!/usr/bin/env python3
"""Prepare one model and run ONNX Runtime or tinygrad on identical bytes."""
import argparse, os, resource, struct, time
from pathlib import Path
import numpy as np

IN,HID,OUT=784,32,10
def idx(path,off):
    b=Path(path).read_bytes(); shape=struct.unpack(">"+"I"*((off-4)//4),b[4:off]); return np.frombuffer(b,dtype=np.uint8,offset=off).copy().reshape(shape)
def prepare(a):
    d,o=Path(a.mnist),Path(a.out);o.mkdir(parents=True,exist_ok=True)
    tx=idx(d/'train-images-idx3-ubyte',16).reshape(-1,IN).astype(np.float32)/255;ty=idx(d/'train-labels-idx1-ubyte',8).reshape(-1)
    ex=idx(d/'t10k-images-idx3-ubyte',16).reshape(-1,IN).astype(np.float32)/255;ey=idx(d/'t10k-labels-idx1-ubyte',8).reshape(-1)
    rng=np.random.default_rng(7);w1=(rng.standard_normal((IN,HID),dtype=np.float32)*.05);b1=np.zeros(HID,np.float32);w2=(rng.standard_normal((HID,OUT),dtype=np.float32)*.1);b2=np.zeros(OUT,np.float32)
    # Preparation is outside every measured path. Mini-batch training only
    # creates a non-toy fixed model shared byte-for-byte by all runners.
    for _ in range(20):
      for s in range(0,10000,100):
        x=tx[s:s+100]; y=np.eye(OUT,dtype=np.float32)[ty[s:s+100]];z=x@w1+b1;h=np.maximum(z,0);pred=h@w2+b2;g=2*(pred-y)/(len(x)*OUT);dh=g@w2.T;dz=dh*(z>0);w2-=.15*(h.T@g);b2-=.15*g.sum(0);w1-=.15*(x.T@dz);b1-=.15*dz.sum(0)
    for n,v in [('w1',w1),('b1',b1),('w2',w2),('b2',b2),('inputs',ex[:1000])]:v.astype('<f4').tofile(o/f'{n}.bin')
    ey[:1000].astype('u1').tofile(o/'labels.bin')
    import onnx
    from onnx import helper,TensorProto,numpy_helper
    nodes=[helper.make_node('MatMul',['x','w1'],['m1']),helper.make_node('Add',['m1','b1'],['a1']),helper.make_node('Relu',['a1'],['h']),helper.make_node('MatMul',['h','w2'],['m2']),helper.make_node('Add',['m2','b2'],['out'])]
    graph=helper.make_graph(nodes,'mnist',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,IN])],[helper.make_tensor_value_info('out',TensorProto.FLOAT,[1,OUT])],[numpy_helper.from_array(w1,'w1'),numpy_helper.from_array(b1,'b1'),numpy_helper.from_array(w2,'w2'),numpy_helper.from_array(b2,'b2')])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=10);onnx.save(model,o/'model.onnx')
def data(d):
    d=Path(d);return tuple(np.fromfile(d/f'{n}.bin',dtype='<f4').reshape(sh) for n,sh in [('w1',(IN,HID)),('b1',(HID,)),('w2',(HID,OUT)),('b2',(OUT,)),('inputs',(1000,IN))])+(np.fromfile(d/'labels.bin',dtype='u1'),)
def emit(engine,elapsed,correct,sink):
    print(f'engine\t{engine}\nwarm_median_ns\t{elapsed:.3f}\naccuracy\t{correct:.6f}\npeak_rss_bytes\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\nchecksum\t{sink:.9g}')
def ort(a):
    import onnxruntime as rt
    w1,b1,w2,b2,x,y=data(a.model);s=rt.InferenceSession(str(Path(a.model)/'model.onnx'),providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name
    xx=x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x; yy=y[:len(xx)];out=np.concatenate([s.run(None,{name:q[None]})[0] for q in xx]);correct=(out.argmax(1)==yy).mean();times=[]
    for r in range(a.reps):t=time.perf_counter_ns();z=s.run(None,{name:x[r%1000:r%1000+1]})[0];times.append(time.perf_counter_ns()-t)
    emit('onnxruntime',float(np.median(times)),correct,float(out.sum()))
def tiny(a):
    from tinygrad import Tensor
    w1,b1,w2,b2,x,y=data(a.model);tw1,tb1,tw2,tb2=map(Tensor,(w1,b1,w2,b2));xx=x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x;yy=y[:len(xx)];out=np.stack([((Tensor(q)@tw1+tb1).relu()@tw2+tb2).numpy() for q in xx]);correct=(out.argmax(1)==yy).mean();times=[]
    for r in range(a.reps):t=time.perf_counter_ns();z=((Tensor(x[r%1000])@tw1+tb1).relu()@tw2+tb2).numpy();times.append(time.perf_counter_ns()-t)
    emit('tinygrad',float(np.median(times)),correct,float(out.sum()))
ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);p=sp.add_parser('prepare');p.add_argument('--mnist',required=True);p.add_argument('--out',required=True);p.set_defaults(fn=prepare)
for n,fn in [('onnxruntime',ort),('tinygrad',tiny)]:p=sp.add_parser(n);p.add_argument('--model',required=True);p.add_argument('--reps',type=int,default=1000);p.set_defaults(fn=fn)
a=ap.parse_args();a.fn(a)
