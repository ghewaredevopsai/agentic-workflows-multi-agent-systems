#!/usr/bin/env python3
"""Geometry check for a deck. Makes the "no slide overflows the stage" rule executable.

    python3 presentation/check-deck-layout.py presentation/module-4-tool-calling-and-mcp.html

The slide runner CLIPS rather than scrolls, so an overflowing slide loses its
bottom silently, and a caption that runs out of its box only looks wrong on a
projector. Neither is visible in the HTML source, and neither survives review by
reading. So render the deck in headless Chrome, force every slide visible, and
measure. Four kinds of problem are reported:

  overflow  the slide is taller than the 720px stage -- the bottom is cut off
  viewBox   SVG text runs outside its own viewBox
  rect      text starts inside a box and runs out of it
  collide   text overlaps a box it does not belong to

Adapted from the equivalent check in the Copilot ADLC course. It found a real
defect on its first run here: Module 3 slide 5, a red caption overrunning into
the right-hand column. Run it after every deck edit.

Needs google-chrome on PATH. No network, no other dependency.
"""
import subprocess, json, re, sys, html, os, tempfile
probe = r"""
(function(){
  document.querySelectorAll('.slide').forEach(function(s){s.style.display='flex';});
  var out=[];
  // real collision only: both axes must overlap substantially. A glyph descending
  // 2-3px past its own box into the parent panel reads fine and is not a defect.
  function ov(a,b){
    var h=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x);
    var v=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
    return (h>15 && v>5) ? Math.round(Math.min(h,v)) : 0;
  }
  document.querySelectorAll('.slide').forEach(function(sl,si){
    sl.querySelectorAll('svg[viewBox]').forEach(function(svg){
      var vb=svg.getAttribute('viewBox').trim().split(/[\s,]+/).map(Number);
      var VW=vb[2],VH=vb[3];
      var rects=[].slice.call(svg.querySelectorAll('rect')).map(function(r){
        var b;try{b=r.getBBox();}catch(e){return null;} return b;}).filter(Boolean);
      svg.querySelectorAll('text').forEach(function(t){
        var b;try{b=t.getBBox();}catch(e){return;}
        if(!b||b.width===0)return;
        if(t.getAttribute('transform'))return;           // rotated: getBBox is un-rotated
        var txt=(t.textContent||'').trim().slice(0,48);
        if(b.x+b.width>VW+0.5||b.y+b.height>VH+0.5||b.x<-0.5){
          out.push({s:si+1,k:'viewBox',t:txt,d:Math.round(Math.max(b.x+b.width-VW,b.y+b.height-VH))});
          return;
        }
        // the container rect: the smallest rect fully containing the text
        var container=null;
        rects.forEach(function(r){
          if(b.x>=r.x-1&&b.x+b.width<=r.x+r.width+1&&b.y>=r.y-1&&b.y+b.height<=r.y+r.height+1)
            if(!container||r.width*r.height<container.width*container.height) container=r;
        });
        // starts inside a rect but runs out of it
        var start=null;
        rects.forEach(function(r){
          if(b.x>=r.x-0.5&&b.x<=r.x+r.width&&b.y-b.height/2>=r.y-2&&b.y<=r.y+r.height+2)
            if(!start||r.width*r.height<start.width*start.height) start=r;
        });
        if(start&&b.x+b.width>start.x+start.width+2){
          out.push({s:si+1,k:'rect',t:txt,d:Math.round(b.x+b.width-(start.x+start.width))});
          return;
        }
        // NEW: overlaps a rect it is not contained by
        for(var i=0;i<rects.length;i++){
          var r=rects[i];
          if(container&&r===container)continue;
          if(start&&r===start)continue;   // the box the text belongs to, even if it descends past it
          if(container&&r.width*r.height>=container.width*container.height)continue;
          var o=ov(b,r);
          if(o){ out.push({s:si+1,k:'collide',t:txt,d:o}); break; }
        }
      });
    });
  });
  document.querySelectorAll('#stage .slide').forEach(function(sl,si){
    if(sl.scrollHeight>721) out.push({s:si+1,k:'overflow',t:(sl.querySelector('h2,h1')||{textContent:''}).textContent.trim().slice(0,48),d:sl.scrollHeight-720});
    sl.style.display='';
  });
  var d=document.createElement('div');d.id='PROBE_RESULT';
  d.textContent=JSON.stringify(out);document.body.appendChild(d);
})();
"""
path=sys.argv[1]
inj=open(path).read().replace('</body>','<script>'+probe+'</script></body>')
fd,tmp=tempfile.mkstemp(suffix='.html',prefix='deckprobe-')
with os.fdopen(fd,'w') as fh: fh.write(inj)
try:
    dom=subprocess.run(['google-chrome','--headless','--disable-gpu','--no-sandbox',
                        '--virtual-time-budget=4000','--dump-dom','file://'+tmp],
                       capture_output=True,text=True).stdout
finally:
    os.unlink(tmp)
m=re.search(r'id="PROBE_RESULT">(.*?)</div>',dom,re.S)
if not m: print('PROBE FAILED'); sys.exit(1)
res=json.loads(html.unescape(m.group(1)))
print('=== %s — %d geometry problems ===' % (path.split('/')[-1], len(res)))
for r in res: print('  slide %-3s %-9s +%-4s %s' % (r['s'],r['k'],r['d'],r['t']))
sys.exit(1 if res else 0)
