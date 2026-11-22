const fs=require('fs'),vm=require('vm'),cp=require('child_process'),assert=require('assert');
const text=fs.readFileSync('dist/app.js','utf8');const a=text.indexOf('function model('),b=text.indexOf('\nconst defs=',a);const context={Math};vm.createContext(context);vm.runInContext(text.slice(a,b),context);
const fixtures=JSON.parse(cp.execFileSync('python3',['-c',`import json
from engine import Scenario,evaluate,capacity
from dataclasses import asdict
fixtures=[]
for s in [Scenario(),Scenario(alpha=0),Scenario(adv=1e6,spread_bps=20),Scenario(days=5),Scenario(turnover=.7)]:
 fixtures.append({'input':asdict(s),'eval':evaluate(s,1e7),'cap':capacity(s)})
print(json.dumps(fixtures))`],{encoding:'utf8'}));
for(const f of fixtures){const s=f.input;const x={alpha:s.alpha,vol:s.daily_vol,adv:s.adv,n:s.names,t:s.turnover,spread:s.spread_bps,fee:s.fee_bps,y:s.impact,p:s.participation,days:s.days,half:s.half_life};const m=context.model(x,1e7);for(const [actual,expected] of [[m.net,f.eval.net],[m.zero,f.cap.break_even],[m.cap,f.cap.usable],[m.opt,f.cap.optimum]])assert(Math.abs(actual-expected)<1e-8*Math.max(1,Math.abs(expected)));}
new vm.Script(text);console.log('Browser formulas match Python for five independent scenarios; JavaScript parses.');
