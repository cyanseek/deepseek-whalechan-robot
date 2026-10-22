"""Local native control panel; sends text, never joint targets."""
import argparse
import io
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk
import urllib.request
from PIL import Image,ImageTk

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=18852);p.add_argument('--profile',choices=['VD25','VD26'],default='VD25');args=p.parse_args()
    endpoint=f'http://127.0.0.1:{args.port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root=tk.Tk();root.title('鲸鱼娘 · '+args.profile+' 具身仿真交互');root.geometry('1100x790')
    picture=ttk.Label(root);picture.pack(fill='both',expand=True)
    status=tk.StringVar(value='连接仿真中…');ttk.Label(root,textvariable=status,font=('Microsoft YaHei',11)).pack(anchor='w',padx=15)
    reply=tk.StringVar();ttk.Label(root,textvariable=reply,font=('Microsoft YaHei',11)).pack(anchor='w',padx=15)
    row=ttk.Frame(root);row.pack(fill='x',padx=15,pady=10)
    text=tk.StringVar();entry=ttk.Entry(row,textvariable=text,font=('Microsoft YaHei',12));entry.pack(side='left',fill='x',expand=True)
    jobs=queue.Queue();updates=queue.Queue(maxsize=2);closed=threading.Event()
    def send(value=None):
        value=value if value is not None else text.get()
        if value.strip():jobs.put(value);text.set('')
    ttk.Button(row,text='发送',command=send).pack(side='left',padx=6);entry.bind('<Return>',lambda e:send())
    buttons=ttk.Frame(root);buttons.pack(fill='x',padx=15,pady=8)
    for label,value in [('过来（红色标记）','过来'),('挥手','挥手'),('摆尾','摆尾'),('停一下','停一下')]+([('小跳','跳一下')] if args.profile=='VD26' else []):
        ttk.Button(buttons,text=label,command=lambda v=value:send(v)).pack(side='left',padx=5)
    ttk.Label(root,text='“过来”使用相机可见红标记。停止会完成收步；运行结束后自动关闭面板。').pack(anchor='w',padx=15,pady=5)
    def worker():
        while not closed.is_set():
            try:
                while not jobs.empty():
                    request=urllib.request.Request(endpoint+'/command',data=json.dumps({'text':jobs.get_nowait()}).encode(),headers={'Content-Type':'application/json'})
                    with opener.open(request,timeout=2) as response:response.read()
                with opener.open(endpoint+'/state',timeout=2) as response:state=json.load(response)
                with opener.open(endpoint+'/frame.jpg',timeout=2) as response:raw=response.read()
                if updates.full():updates.get_nowait()
                updates.put_nowait((state,raw,None))
            except Exception:
                if not updates.full():updates.put_nowait((None,None,'仿真尚未连接，或本次运行已结束。'))
            closed.wait(.10)
    def refresh():
        while not updates.empty():
            state,raw,error=updates.get_nowait()
            if error:status.set(error)
            else:
                names={'idle':'等待指令','thinking':'正在理解','waiting_for_support':'先站稳，再执行','approach':'走近标记',
                       'wave':'挥手','tail':'摆尾','jump':'准备小跳与落地','stopping':'正在收步','stopped':'已经停稳','complete':'动作完成',
                       'arrived':'到达标记附近','settling_at_target':'正在站稳','lost':'目标丢失','reply':'回应','model_error':'请重试'}
                backend='DeepSeek 已接入' if state.get('backend')=='deepseek' else '离线指令模式'
                status.set(f"{backend} | {names.get(state.get('status'),state.get('status',''))} | 仿真 {state.get('sim_s',0):.1f}s")
                reply.set('鲸鱼娘：'+state.get('reply',''))
                if raw:
                    im=Image.open(io.BytesIO(raw));im.thumbnail((1080,607));photo=ImageTk.PhotoImage(im)
                    picture.configure(image=photo);picture.image=photo
        root.after(80,refresh)
    root.protocol('WM_DELETE_WINDOW',lambda:(closed.set(),root.destroy()))
    threading.Thread(target=worker,daemon=True).start();refresh();entry.focus_set();root.mainloop()

if __name__=='__main__':main()
