"""Silent schematic workflow plus measured masking-sweep results; no real-time telemetry."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
import imageio_ffmpeg
from experiment import ROOT
from masking_sweep import FRACTIONS,result_matrix

FPS=12
SECONDS=20

def draw_frame(matrix,labels,frame):
    t=frame/FPS;fig=plt.figure(figsize=(10.8,10.8),dpi=100,facecolor='white')
    ax=fig.add_axes([.03,.02,.94,.96]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    navy='#183346';teal='#087F8C';light='#EEF4F7';amber='#D39332'
    nodes=[('Past reports',.02),('Draw delay',.265),('Causal features',.51),('HGB: 150 trees',.755)]
    active=min(3,int(t//2))
    for i,(text,x) in enumerate(nodes):
        color=teal if i==active and t<8 else navy
        ax.add_patch(FancyBboxPatch((x,.84),.22,.065,boxstyle='round,pad=.007',facecolor=light,edgecolor=color,linewidth=2.2))
        ax.text(x+.11,.872,text,ha='center',va='center',color=color,fontsize=15)
        if i<3:ax.add_patch(FancyArrowPatch((x+.225,.872),(nodes[i+1][1]-.01,.872),arrowstyle='-|>',mutation_scale=16,color=teal,linewidth=2))
    ax.text(.5,.78,r'$P(d=0)=1-q\quad P(d=1)=P(d=3)=P(d=6)=q/3$',ha='center',fontsize=16,color=navy)
    ax.text(.5,.738,'Same training rows • causal lag rebuild • no target masking',ha='center',fontsize=15,color=navy)
    ax.text(.5,.688,'Train < test; delays 0/1/3/6h; next-hour forecasts',ha='center',fontsize=15,color=teal)
    table_ax=fig.add_axes([.035,.23,.93,.40]);table_ax.axis('off')
    # Sequential reveal is a schematic presentation order, not training latency.
    revealed=0 if t<8 else min(len(labels),int((t-8)//.6)+1)
    table_data=[]
    for i,label in enumerate(labels):table_data.append([label]+[f'{matrix.iloc[i,j]:.1f}' if i<revealed else '—' for j in range(4)])
    table=table_ax.table(cellText=table_data,colLabels=['MAE (rentals/hour)','0h','1h','3h','6h'],cellLoc='center',loc='center',colWidths=[.42,.145,.145,.145,.145])
    table.auto_set_font_size(False);table.set_fontsize(16);table.scale(1,2.5)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#CBDCE3');cell.set_facecolor(light if i==0 else 'white');cell.get_text().set_color(navy)
    ax.text(.5,.20,'Lower MAE is better; all five preset recipes retained',ha='center',fontsize=15,color=navy)
    ax.text(.5,.147,'UCI Bike Sharing; 2,208 target hours; July–September 2012',ha='center',fontsize=14,color=navy)
    ax.text(.5,.11,'18 fitted models; mask seeds 42/43/44; one draw per fold',ha='center',fontsize=14,color=navy)
    ax.text(.5,.074,'Exploratory benchmark reuse; no independent validation',ha='center',fontsize=14,color=navy)
    ax.text(.5,.042,'Schematic workflow: animation timing is not measured training time',ha='center',fontsize=12,color=navy)
    ax.text(.5,.015,'Data: doi.org/10.24432/C5W894 • CC BY 4.0 • silent',ha='center',fontsize=12,color=navy)
    fig.canvas.draw();rgba=np.asarray(fig.canvas.buffer_rgba());rgb=rgba[:,:,:3].copy();plt.close(fig)
    return rgb

def render(directory):
    metrics=pd.read_csv(directory/'metrics.csv');matrix,labels=result_matrix(metrics,FRACTIONS)
    outfile=directory/'masking-workflow.mp4'
    writer=imageio_ffmpeg.write_frames(str(outfile),(1080,1080),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
        quality=8,macro_block_size=1,ffmpeg_log_level='error',output_params=['-movflags','+faststart'])
    writer.send(None)
    cached={}
    for n in range(FPS*SECONDS):
        # Reuse identical presentation states to avoid redundant chart rendering.
        key=(min(3,int((n/FPS)//2)),0 if n/FPS<8 else min(7,int((n/FPS-8)//.6)+1))
        if key not in cached:cached[key]=draw_frame(matrix,labels,n)
        rgb=cached[key];writer.send(rgb)
    writer.close()
    reader=imageio_ffmpeg.read_frames(str(outfile),pix_fmt='rgb24');meta=next(reader);count=0
    saved=[]
    for n,raw in enumerate(reader):
        count+=1
        if n in (0,120,239):
            frame=np.frombuffer(raw,dtype=np.uint8).reshape(meta['size'][1],meta['size'][0],3)
            path=directory/f'workflow-frame-{n:03d}.png';Image.fromarray(frame).save(path);saved.append(path.name)
    reader.close();assert count==FPS*SECONDS and tuple(meta['size'])==(1080,1080) and abs(meta['duration']-SECONDS)<.1
    assert meta['codec']=='h264' and abs(meta['fps']-FPS)<.01
    result={'file':outfile.name,'sha256':hashlib.sha256(outfile.read_bytes()).hexdigest(),'width':1080,'height':1080,'fps':FPS,
        'decoded_frames':count,'duration_seconds':meta['duration'],'codec':meta['codec'],'silent':True,
        'type':'Schematic technical workflow plus actual measured result table; presentation speed is not measured latency',
        'first_middle_last_frames':saved,'all_frames_decoded':True,'numerical_source':'metrics.csv',
        'renderer_version':importlib.metadata.version('imageio-ffmpeg'),'ffmpeg_version':meta['ffmpeg_version'],
        'browser_playback_check':'Unavailable: connected Chrome absent; full decode verified. Publisher must inspect native uploaded video.',
        'command':'python render_masking_workflow.py --directory experiments/2026-10-13'}
    (directory/'video-checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('file','width','height','decoded_frames','duration_seconds','codec','all_frames_decoded')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=ROOT/'experiments'/'2026-10-13');args=parser.parse_args();render(args.directory)
