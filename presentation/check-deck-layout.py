#!/usr/bin/env python3
"""Geometry check for a deck. Makes the "no slide overflows the stage" rule executable.

    python3 presentation/check-deck-layout.py presentation/module-4-tool-calling-and-mcp.html
    python3 presentation/check-deck-layout.py <deck> --baseline <older copy of the same deck>

The slide runner CLIPS rather than scrolls, so an overflowing slide loses its
bottom silently, and a caption that runs out of its box only looks wrong on a
projector. Neither is visible in the HTML source, and neither survives review by
reading. So render the deck in headless Chrome, force every slide visible, and
measure. Five kinds of problem fail the run:

  overflow  the slide is taller than the 720px stage -- the bottom is cut off
  viewBox   SVG text runs outside its own viewBox
  rect      text starts inside a box and runs out of it
  collide   text overlaps a box it does not belong to
  textovl   two pieces of text overlap each other

...and one is advisory, printed only with --baseline (see below):

  onpath    text overlaps a line/path/circle rather than a box

Adapted from the equivalent check in the Copilot ADLC course. It found a real
defect on its first run here: Module 3 slide 5, a red caption overrunning into
the right-hand column. Run it after every deck edit.

WHY textovl EXISTS: the box checks above compare text against <rect> only, so two
captions could sit on top of each other and still report a clean bill of health.
Raising the deck type scale on 2026-09-07 did exactly that -- Module 2 slide 8's
"score every branch" landed on "C pruned", and Module 4 slide 8 had two more that
had been shipping unnoticed. Text-vs-text is a hard failure: across all nine decks
it has no false positives.

WHY onpath IS ADVISORY: a label sitting on a line is usually deliberate -- a number
inside a circle, a "yes"/"no" beside an arrow. There are ~44 such legitimate cases
across the decks, so failing on them would be useless noise. It earns its keep as a
DIFF instead: after a resize, run with --baseline pointing at the pre-change copy
(git show HEAD:path > /tmp/old.html) and only genuinely new collisions are printed.
That is how the Module 4 chart was caught, where the axis title had come to rest on
top of the trend curve.

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
      // everything that is NOT a box, for the advisory onpath check
      var geo=[].slice.call(svg.querySelectorAll('path,line,circle,polyline,polygon')).map(function(g){
        var b;try{b=g.getBBox();}catch(e){return null;}
        return (b&&(b.width||b.height))?b:null;}).filter(Boolean);
      // untransformed text, collected for the pairwise checks below
      var texts=[];
      svg.querySelectorAll('text').forEach(function(t){
        var b;try{b=t.getBBox();}catch(e){return;}
        if(!b||b.width===0)return;
        var txt=(t.textContent||'').trim().slice(0,48);
        var tf=t.getAttribute('transform');
        if(tf){
          // getBBox is pre-transform. For rotate(a cx cy) -- the only form these decks use --
          // map the four corners and take the axis-aligned hull, then viewBox-check that.
          // Rotated captions clip silently otherwise; one did, and only a screenshot caught it.
          var m=/rotate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)/.exec(tf);
          if(!m)return;                                  // any other transform: skip, as before
          var a=+m[1]*Math.PI/180, cx=+m[2], cy=+m[3], xs=[], ys=[];
          [[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]]
            .forEach(function(p){
              var dx=p[0]-cx, dy=p[1]-cy;
              xs.push(cx+dx*Math.cos(a)-dy*Math.sin(a));
              ys.push(cy+dx*Math.sin(a)+dy*Math.cos(a));
            });
          var rx=Math.min.apply(null,xs), ry=Math.min.apply(null,ys);
          var rw=Math.max.apply(null,xs)-rx, rh=Math.max.apply(null,ys)-ry;
          if(rx+rw>VW+0.5||ry+rh>VH+0.5||rx<-0.5||ry<-0.5)
            out.push({s:si+1,k:'rotated',t:txt,
                      d:Math.round(Math.max(rx+rw-VW,ry+rh-VH,-rx,-ry))});
          return;                                        // rect collisions: still not checked
        }
        texts.push({b:b,t:txt});
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
      // text vs text. The rect checks above never compare two captions, so a pair
      // sitting on top of each other passed review until this was added.
      for(var i=0;i<texts.length;i++)for(var j=i+1;j<texts.length;j++){
        var a=texts[i].b, c=texts[j].b;
        var h=Math.min(a.x+a.width,c.x+c.width)-Math.max(a.x,c.x);
        var v=Math.min(a.y+a.height,c.y+c.height)-Math.max(a.y,c.y);
        if(h>4&&v>3)
          out.push({s:si+1,k:'textovl',d:Math.round(Math.min(h,v)),
                    t:texts[i].t+'  ||  '+texts[j].t});
      }
      // text vs line/path/circle. Advisory -- see the module docstring.
      texts.forEach(function(x){
        var worst=0;
        geo.forEach(function(g){
          var h=Math.min(x.b.x+x.b.width,g.x+g.width)-Math.max(x.b.x,g.x);
          var v=Math.min(x.b.y+x.b.height,g.y+g.height)-Math.max(x.b.y,g.y);
          if(h>6&&v>6) worst=Math.max(worst,Math.round(Math.min(h,v)));
        });
        if(worst) out.push({s:si+1,k:'onpath',t:x.t,d:worst});
      });
    });
  });
  document.querySelectorAll('#stage .slide').forEach(function(sl,si){
    if(sl.scrollHeight>721) out.push({s:si+1,k:'overflow',t:(sl.querySelector('h2,h1')||{textContent:''}).textContent.trim().slice(0,48),d:sl.scrollHeight-720});
    sl.style.display='';
  });
  // A deck whose template never closed renders no slides at all, and would otherwise
  // report a clean bill of health. Count what was actually inspected.
  var seen=document.querySelectorAll('#stage .slide').length;
  if(seen===0) out.push({s:0,k:'NOSLIDES',t:'the runner rendered no slides -- is the markup intact?',d:0});
  var d=document.createElement('div');d.id='PROBE_RESULT';
  d.textContent=JSON.stringify(out);document.body.appendChild(d);
})();
"""

def measure(path):
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
    if not m: print('PROBE FAILED for '+path); sys.exit(1)
    return json.loads(html.unescape(m.group(1)))

args=[a for a in sys.argv[1:] if a!='--baseline']
baseline = sys.argv[sys.argv.index('--baseline')+1] if '--baseline' in sys.argv else None
path=args[0]
if baseline and baseline in args: args.remove(baseline)

res=measure(path)
hard=[r for r in res if r['k']!='onpath']
soft=[r for r in res if r['k']=='onpath']

print('=== %s — %d geometry problems ===' % (path.split('/')[-1], len(hard)))
for r in hard: print('  slide %-3s %-9s +%-4s %s' % (r['s'],r['k'],r['d'],r['t']))

if baseline:
    # only report label-on-line collisions this edit INTRODUCED; the deliberate ones
    # (a number in a circle, "yes" beside an arrow) are in the baseline too and drop out.
    was={(r['s'],r['t']) for r in measure(baseline) if r['k']=='onpath'}
    new=[r for r in soft if (r['s'],r['t']) not in was]
    print('--- %d new text-on-line overlaps vs %s (advisory) ---'
          % (len(new), baseline.split('/')[-1]))
    for r in new: print('  slide %-3s %-9s +%-4s %s' % (r['s'],r['k'],r['d'],r['t']))

sys.exit(1 if hard else 0)
