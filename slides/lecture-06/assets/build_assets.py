"""Build editable diagrams and Plotly figures from local Lecture 06 evidence.

Usage: uv run python slides/lecture-06/assets/build_assets.py --evidence PATH
No training runs or remote downloads are performed here. SVG and Excalidraw
share the same coordinates and text, following the Lecture 01 diagram builder.
"""
import argparse
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAVY, BLUE, GREEN, PURPLE, MUTED = "#142e4b", "#20578c", "#28673f", "#634185", "#566471"


class Drawing:
    def __init__(self, name, title):
        self.name, self.title = name, title
        self.elements, self.svg = [], []

    def element(self, kind, x, y, w, h, color=NAVY, fill="transparent"):
        result = dict(id=f"{self.name}-{len(self.elements)}", type=kind, x=x, y=y,
                      width=w, height=h, angle=0, strokeColor=color,
                      backgroundColor=fill, fillStyle="solid", strokeWidth=2,
                      strokeStyle="solid", roughness=0, opacity=100, groupIds=[],
                      frameId=None, roundness=None, seed=400+len(self.elements),
                      version=1, versionNonce=600+len(self.elements), isDeleted=False,
                      boundElements=None, updated=1, link=None, locked=False)
        self.elements.append(result)
        return result

    def text(self, x, y, value, width=240, size=32, color=NAVY, align="center"):
        lines = value.split("\n")
        self.element("text",x,y,width,len(lines)*size*1.25,color).update(
            fontSize=size,fontFamily=2,text=value,originalText=value,textAlign=align,
            verticalAlign="top",containerId=None,autoResize=False,lineHeight=1.25)
        for i,line in enumerate(lines):
            anchor="middle" if align=="center" else "start"
            xx=x+width/2 if align=="center" else x
            self.svg.append(f'<text x="{xx}" y="{y+size+i*size*1.25}" font-size="{size}" '
                            f'text-anchor="{anchor}" fill="{color}">{escape(line)}</text>')

    def box(self,x,y,w,h,label,color=BLUE,size=32):
        self.element("rectangle",x,y,w,h,color,"#eef2f5")
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                        f'fill="#eef2f5" stroke="{color}" stroke-width="2"/>')
        self.text(x+8,y+(h-len(label.split(chr(10)))*size*1.25)/2-2,label,w-16,size,color)

    def arrow(self,x1,y1,x2,y2,color=BLUE):
        left, top = min(x1,x2), min(y1,y2)
        self.element("arrow",left,top,abs(x2-x1),abs(y2-y1),color).update(
            points=[[x1-left,y1-top],[x2-left,y2-top]],lastCommittedPoint=None,startBinding=None,
            endBinding=None,startArrowhead=None,endArrowhead="arrow",elbowed=False)
        self.svg.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" '
                        f'stroke-width="3" marker-end="url(#arrow)"/>')

    def save(self):
        scene=dict(type="excalidraw",version=2,source="https://excalidraw.com",
                   elements=self.elements,appState={"viewBackgroundColor":"#fbfbf9","gridSize":None},files={})
        (HERE/f"{self.name}.excalidraw").write_text(json.dumps(scene,indent=2)+"\n")
        (HERE/f"{self.name}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="410" viewBox="0 0 1152 410" role="img">'
            f'<title>{escape(self.title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#20578c"/></marker></defs>'
            '<g font-family="Arial, Helvetica, sans-serif">'+''.join(self.svg)+'</g></svg>\n')


def diagrams():
    d=Drawing("noa-model","The exact 4608-parameter Noa teaching model")
    for x,w,label in [(10,245,"Token + position\n12 × 16 + 10 × 16"),(315,250,"2 pre-LN blocks\n4 heads · FFN 32"),
                      (625,200,"Final norm\nwidth 16"),(885,250,"Tied readout\n16 → 12 logits")]:
        d.box(x,85,w,130,label,size=30)
    for x1,x2 in [(258,302),(568,612),(828,872)]:d.arrow(x1,150,x2,150)
    d.text(20,20,"Noa can be annoying but she is a great cat EOS",1110,34)
    d.text(25,270,"4,608 unique parameters · context limit: 10 inputs",1100,34,color=GREEN)
    d.text(25,332,"ReLU · no Linear biases · no dropout · tied token/readout weights",1100,28,color=MUTED)
    d.save()

    d=Drawing("shifted-targets","Each input position predicts the following token")
    d.text(0,24,"Input",175,32,BLUE)
    d.text(0,230,"Target",175,32,GREEN)
    for i,(a,b) in enumerate(zip(["Noa","can","be","annoying"],["can","be","annoying","but"])):
        x=188+i*235
        d.box(x,18,214,72,a,size=32)
        d.arrow(x+107,109,x+107,213)
        d.box(x,231,214,72,b,color=GREEN,size=32)
    d.text(15,341,"Four scored positions; each prediction uses only its input prefix.",1120,30)
    d.save()

    d=Drawing("training-update","One optimizer update in eight ordered stages")
    labels=["1  Train mode\nclear gradients","2  Forward\nraw logits","3  Mean loss\nshifted targets","4  Backward\nparameter grads",
            "5  Clip once\nglobal norm","6  Set LR\nfor this update","7  Optimizer\nchanges weights","8  Record\ncompleted update"]
    for row in range(2):
        for col in range(4):
            x,y=10+col*290,20+row*185
            d.box(x,y,260,130,labels[row*4+col],size=30)
            if col<3:d.arrow(x+265,y+65,x+282,y+65)
    d.text(20,355,"Accumulate microbatches before stages 5–8; clear once per update.",1100,28,color=MUTED)
    d.save()

    d=Drawing("checkpoint-state","The state needed to continue the next update")
    d.text(5,4,"Training state",410,34)
    for y,label in [(60,"Model parameters"),(136,"AdamW moments + step"),(212,"Completed updates + config"),(288,"RNG + sampler + data IDs")]:
        d.box(5,y,435,60,label,size=30)
    d.arrow(458,200,590,200)
    d.box(610,115,235,175,"Save after\na complete\nupdate",size=32)
    d.arrow(860,200,955,200)
    d.text(958,125,"Next batch\nNext LR\nNext update",190,30,align="left")
    d.save()

    d=Drawing("resume-timeline","Compare 40 continuous updates with a fresh-process restart")
    d.text(0,2,"0 updates",270,30)
    d.text(440,2,"20 updates",270,30)
    d.text(882,2,"40 updates",270,30)
    for y,title in [(105,"Continuous"),(245,"Restarted")]:
        d.text(5,y-42,title,240,32,align="left")
        d.arrow(30,y+18,1110,y+18)
    d.box(465,182,220,126,"Save\nnew process\nrestore",color=GREEN,size=28)
    d.text(10,342,"Same initial state, future batches, LR schedule, and CPU environment.",1130,30)
    d.save()

    d=Drawing("generation-loop","Use final-position logits to append a token, then repeat")
    d.text(10,8,"Prefix: Noa can be annoying but she is a great",1130,34)
    for x,w,label in [(10,255,"Causal model\nlast-position logits"),(340,205,"Choose\na token"),(620,200,"Append\ncat"),(895,245,"Next prefix\nrepeat")]:
        d.box(x,126,w,130,label,size=30)
    for a,b in [(273,327),(550,607),(830,882)]:d.arrow(a,190,b,190)
    d.text(15,325,"Stop at EOS or the token budget. The example token is illustrative.",1120,30,color=MUTED)
    d.save()

    d=Drawing("greedy-tree","A fixed-length two-step probability tree")
    d.box(0,147,140,92,"Prefix",size=30)
    d.box(350,45,175,82,"A: 0.6",size=32)
    d.box(350,262,175,82,"B: 0.4",size=32)
    d.arrow(153,177,332,88)
    d.arrow(153,219,332,300)
    for y,label in [(10,"X: 0.5 → joint 0.30"),(113,"Y: 0.5 → joint 0.30"),
                    (234,"X: 0.9 → joint 0.36"),(337,"Y: 0.1 → joint 0.04")]:
        d.box(710,y,425, 60,label,color=GREEN if y==234 else BLUE,size=30)
    for y,ends in [(86,[40,143]),(303,[264,367])]:
        for end in ends:d.arrow(540,y,695,end)
    d.save()


def plot(name,data,layout=None):
    value={"data":data,"layout":layout or {},"config":{"displayModeBar":False}}
    (HERE/f"{name}.json").write_text(json.dumps(value,indent=2)+"\n")


def charts(e):
    from training import lr_at
    plot("learning-rate",[{"x":list(range(40)),"y":[lr_at(i) for i in range(40)],"mode":"lines+markers","name":"LR"}],
         {"xaxis":{"title":{"text":"Optimizer update s (zero-based)"}},"yaxis":{"title":{"text":"Learning rate"},"rangemode":"tozero"}})
    fit=e["fit"]["curve"]
    plot("tiny-fit",[{"x":[r["completed"] for r in fit],"y":[r["nll"] for r in fit],"name":"Noa full batch","mode":"lines"},
                     {"x":[0,200],"y":[.06931471805599453]*2,"name":"ln(2)/10 floor","mode":"lines","line":{"dash":"dash"}}],
         {"xaxis":{"title":{"text":"Completed optimizer updates"}},"yaxis":{"title":{"text":"Mean NLL (nats/token)"},"type":"log",
          "tickmode":"array","tickvals":[.07,.1,.3,1,3],"ticktext":["0.07","0.1","0.3","1","3"]},
          "legend":{"orientation":"h","x":0,"y":1.17},"margin":{"l":115,"r":35,"t":65,"b":80}})
    curve=e["development"]["curve"]
    plot("development-loss",[{"x":[r["completed"] for r in curve],"y":[r[part]["nll"] for r in curve],"name":part,"mode":"lines+markers"} for part in ["train","dev"]],
         {"xaxis":{"title":{"text":"Completed optimizer updates"}},"yaxis":{"title":{"text":"Mean NLL (nats/token)"},"rangemode":"tozero"},
          "legend":{"orientation":"h","x":0,"y":1.17},"margin":{"l":115,"r":35,"t":65,"b":80}})
    steps=e["development"]["steps"]
    plot("gradient-norm",[{"x":[r["update"] for r in steps],"y":[r["grad_norm"] for r in steps],"name":"Before clipping","mode":"lines"},
                          {"x":[0,199],"y":[1,1],"name":"Clip threshold","mode":"lines","line":{"dash":"dash"}}],
         {"xaxis":{"title":{"text":"Optimizer update s (zero-based)"}},"yaxis":{"title":{"text":"Global L2 gradient norm"}},
          "legend":{"orientation":"h","x":0,"y":1.17},"margin":{"l":115,"r":35,"t":65,"b":80}})
    plot("sampling",[{"x":["A","B","C","D"],"y":[.4,.3,.2,.1],"name":"Temperature only","type":"bar"},
                     {"x":["A","B","C","D"],"y":[4/7,3/7,0,0],"name":"After top-p","type":"bar"}],
         {"barmode":"group","xaxis":{"title":{"text":"Toy token"}},"yaxis":{"title":{"text":"Probability"},"range":[0,1]},
          "legend":{"orientation":"h","x":0,"y":1.17},"margin":{"l":100,"r":35,"t":60,"b":65}})
    plot("temperature",[{"x":["A","B","C","D"],"y":[v**(1/t)/sum(w**(1/t) for w in [.4,.3,.2,.1]) for v in [.4,.3,.2,.1]],
                         "name":f"τ = {t}","type":"bar"} for t in [.5,1,2]],
         {"barmode":"group","xaxis":{"title":{"text":"Toy token"}},"yaxis":{"title":{"text":"Probability"},"range":[0,.6]},
          "legend":{"orientation":"h","x":0,"y":1.17},"margin":{"l":100,"r":35,"t":60,"b":65}})
    # This file is a compact provenance and teaching summary, not the raw run log.
    summary={k:e[k] for k in ["recorded_utc","revision","working_tree","sources_sha256","python","torch","platform","cpu_threads","parameters","sampling"]}
    summary["fit"]={k:v for k,v in e["fit"].items() if k!="curve"}
    summary["fit"]["initial_nll"],summary["fit"]["final_nll"]=fit[0]["nll"],fit[-1]["nll"]
    summary["development"]={k:v for k,v in e["development"].items() if k not in ["curve","steps"]}
    summary["development"].update(initial=curve[0],final=curve[-1])
    summary["resume"]={k:v for k,v in e["resume"].items() if k!="worker_records"}
    summary["resume"]["worker_timings"]={k:{f:v[f] for f in ["run_seconds","process_peak_rss_mib"]} for k,v in e["resume"]["worker_records"].items()}
    (HERE/"evidence-summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    args=parser.parse_args()
    diagrams()
    charts(json.loads(args.evidence.read_text()))
