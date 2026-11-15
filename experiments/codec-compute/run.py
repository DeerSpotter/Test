"""Codec remapping: software PNG arithmetic proof and optional strict NVIDIA decoder probe."""
import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

def execute(args):
    t=time.perf_counter()
    p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(f"Command exit {p.returncode}: {' '.join(map(str,args))}\n{p.stdout}\n{p.stderr}")
    return round((time.perf_counter()-t)*1000,3),p.stdout+p.stderr

def chunk(tag,payload):
    buf=tag+payload
    return struct.pack(">I",len(payload))+buf+struct.pack(">I",zlib.crc32(buf)&0xffffffff)

def cpu():
    n=4096
    a=bytes((i*31+7)&255 for i in range(n))
    b=bytes((i*17+199)&255 for i in range(n))
    expected=bytes((x+y)&255 for x,y in zip(a,b))
    header=struct.pack(">IIBBBBB",n,2,8,0,0,0,0)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',zlib.compress(b'\x00'+a+b'\x02'+b))+chunk(b'IEND',b'')
    with tempfile.TemporaryDirectory() as d:
        inp,out=Path(d)/'a.png',Path(d)/'pixels.raw'
        inp.write_bytes(png)
        ms,_=execute(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(inp),'-frames:v','1','-pix_fmt','gray','-f','rawvideo',str(out)])
        actual=out.read_bytes()
    if actual!=a+expected:
        raise AssertionError(f'PNG Up-filter arithmetic mismatch: decoded {len(actual)} bytes')
    return {'status':'passed','mode':'cpu','operations':n,'accuracy':'exact','wall_ms_including_startup':ms,'decoder':'CPU software PNG','hardware_acceleration':False}

def nvdec():
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('No NVIDIA GPU driver: nvidia-smi is missing')
    _,gpu=execute(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'])
    _,decoders=execute(['ffmpeg','-hide_banner','-decoders'])
    if not re.search(r'\bh264_cuvid\b',decoders):
        raise RuntimeError('FFmpeg lacks h264_cuvid; hardware decoding required, CPU fallback forbidden')
    with tempfile.TemporaryDirectory() as d:
        src=str(Path(d)/'synthetic.mkv')
        execute(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=size=320x240:rate=30','-frames:v','120','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-qp','0',src])
        sw,_=execute(['ffmpeg','-hide_banner','-loglevel','error','-c:v','h264','-i',src,'-f','null','-'])
        hw,_=execute(['ffmpeg','-hide_banner','-loglevel','error','-c:v','h264_cuvid','-i',src,'-f','null','-'])
    return {'status':'passed','mode':'nvdec','gpu':gpu.strip(),'frames':120,'software_decode_ms':sw,'hardware_decode_ms':hw,'limitation':'Hardware VIDEO decode only, not yet non-video arithmetic'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['cpu','nvdec'],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg executable missing')
        result=cpu() if a.mode=='cpu' else nvdec()
    except Exception as e:
        a.out.write_text(json.dumps({'status':'failed','error':str(e),'mode':a.mode},indent=2)+'\n')
        raise
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
