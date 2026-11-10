/* Renderer only. Collisions, score, pause and outcome come from OfflineStars. */
window.starView=(()=>{
const c=document.getElementById('game'),g=c.getContext('2d');let data=null,send=()=>{},x=160;
const laneX=[66,160,254];
function rr(x,y,w,h,r,fill){g.fillStyle=fill;g.beginPath();g.roundRect(x,y,w,h,r);g.fill()}
function text(s,x,y,size=12,color='#315b84'){g.fillStyle=color;g.textAlign='center';g.font=`600 ${size}px "Microsoft YaHei",sans-serif`;g.fillText(s,x,y)}
function star(x,y,r=9){g.fillStyle='#ffd669';g.strokeStyle='#d9a044';g.lineWidth=.7;g.beginPath();for(let i=0;i<10;i++){const a=i*Math.PI/5-Math.PI/2,rr=i%2?r*.46:r;let xx=x+Math.cos(a)*rr,yy=y+Math.sin(a)*rr;i?g.lineTo(xx,yy):g.moveTo(xx,yy)}g.closePath();g.fill();g.stroke()}
function jelly(x,y){g.fillStyle='#aa91db';g.beginPath();g.arc(x,y,10,Math.PI,0);g.lineTo(x+10,y+4);g.lineTo(x-10,y+4);g.fill();g.strokeStyle='#9475c4';g.lineWidth=2;for(let i=-6;i<=6;i+=6){g.beginPath();g.moveTo(x+i,y+4);g.quadraticCurveTo(x+i-4,y+13,x+i+2,y+14);g.stroke()}g.fillStyle='#574e83';g.fillRect(x-5,y-1,2,2);g.fillRect(x+3,y-1,2,2)}
function whale(x,y,t){g.save();g.translate(x,y);g.rotate(Math.sin(t*2)*.035);g.fillStyle='#75bde9';g.beginPath();g.moveTo(21,2);g.quadraticCurveTo(34,-15,38,-11);g.quadraticCurveTo(41,3,26,9);g.fill();g.beginPath();g.ellipse(0,0,25,16,0,0,Math.PI*2);g.fill();g.fillStyle='#eaf9ff';g.beginPath();g.ellipse(-2,7,19,8,0,0,Math.PI);g.fill();g.fillStyle='#3d5c96';g.beginPath();g.moveTo(-20,-8);g.quadraticCurveTo(-3,-25,15,-10);g.lineTo(7,-3);g.lineTo(1,-11);g.lineTo(-7,-3);g.closePath();g.fill();for(let k=-2;k<=2;k++){g.fillStyle='#fff';g.beginPath();g.ellipse(k*6,-17+Math.abs(k)*1.6,4,3.5,-k*.2,0,Math.PI*2);g.fill()}g.fillStyle='#284a76';g.beginPath();g.ellipse(-10,1,2,3,0,0,7);g.fill();g.beginPath();g.ellipse(5,1,2,3,0,0,7);g.fill();g.fillStyle='#edb5c5';g.beginPath();g.ellipse(-16,6,4,2,0,0,7);g.fill();g.beginPath();g.ellipse(12,6,4,2,0,0,7);g.fill();g.strokeStyle='#456c96';g.lineWidth=1;g.beginPath();g.arc(-2,5,3,0,Math.PI);g.stroke();g.restore()}
function draw(ms){requestAnimationFrame(draw);if(c.hidden)return;const s=data?.game,t=ms/1000;g.clearRect(0,0,320,240);const bg=g.createLinearGradient(0,0,0,240);bg.addColorStop(0,'#edfaff');bg.addColorStop(1,'#a9dcf0');g.fillStyle=bg;g.fillRect(0,0,320,240);
for(let i=0;i<12;i++){let y=(230-i*21-(t*7)%252+252)%252;g.strokeStyle='#ffffff88';g.lineWidth=1;g.beginPath();g.arc((i*71+13)%320,y,2+i%4,0,7);g.stroke()}
for(let i=0;i<3;i++){rr(laneX[i]-40,42,80,174,20,i===(s?.lane??1)?'#ffffff44':'#ffffff18');g.fillStyle='#6cadd344';g.fillRect(laneX[i]-27,215,54,2)}
g.fillStyle='#7fc6df55';g.beginPath();g.moveTo(0,230);for(let xx=0;xx<=320;xx+=8)g.lineTo(xx,228+Math.sin(xx*.05+t)*4);g.lineTo(320,240);g.lineTo(0,240);g.fill();
rr(9,8,100,27,11,'#ffffffd9');star(23,21,7);text(`${s?.score??0} / ${s?.goal??6}`,65,26,13);
rr(117,8,85,27,11,'#ffffffd9');text(`${Math.max(0,Math.ceil((s?.duration??35)-(s?.time??0)))} 秒`,160,26,12);
rr(210,8,101,27,11,'#ffffffd9');text('♥'.repeat(s?.lives??3)+'♡'.repeat(3-(s?.lives??3)),260,26,16,'#d98fa8');
for(const o of s?.objects??[]){o.kind==='star'?star(laneX[o.lane],o.y):jelly(laneX[o.lane],o.y)}
x+=(laneX[s?.lane??1]-x)*.24;whale(x,195,t);
const paused=data&&(data.stopped||data.carried||data.state.paused),phase=s?.phase??'ready';
if(paused||phase!=='running'){rr(31,67,258,94,18,'#ffffffec');let title='星海收集站',line='接住星星，绕开小水母',last='点击下方「开始游戏」';if(phase==='won'){title='星星收齐啦！';line='主人，我们把星星捞回来啦';last='再来一局也可以呀'}else if(phase==='lost'){title='再捞一次，好不好？';line=s.lives<=0?'被小水母挠到啦':'时间到，下次再多捞一点';last='点击下方「再来一局」'}if(paused){title=data.carried?'被主人抱起来啦':'先歇一小会儿';line='进度和星星都留在这里';last=data.carried?'放下后按「继续」':'空格 / 暂停继续按钮'}text(title,160,94,19);text(line,160,119,12);text(last,160,145,10,'#6b90ad')}
}requestAnimationFrame(draw);
c.addEventListener('pointerdown',e=>{if(!data?.game||data.game.phase!=='running')return;const b=c.getBoundingClientRect();send('lane',Math.max(0,Math.min(2,Math.floor((e.clientX-b.left)/b.width*3))));c.focus()});
document.addEventListener('keydown',e=>{if(c.hidden||e.target.closest('input,textarea,select,button,a,[contenteditable=true]')||e.repeat)return;const a=({ArrowLeft:'left',a:'left',A:'left',ArrowRight:'right',d:'right',D:'right',' ':'pause'})[e.key];if(a){e.preventDefault();send(a)}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&data?.game?.phase==='running'&&!data.stopped)send('pause')});
return{connect:f=>send=f,update:s=>{data=s;const v=s.game;document.getElementById('gameStatus').textContent=v?`星星 ${v.score} / ${v.goal} · 爱心 ${v.lives} · ${Math.max(0,Math.ceil(v.duration-v.time))} 秒 · ${{ready:'等待开始',running:s.stopped||s.state.paused?'暂停中':'游戏中',won:'通关！',lost:'这局结束'}[v.phase]}`:'先按开始游戏，载入星海收集站';}};
})();
