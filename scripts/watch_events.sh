#!/usr/bin/env bash
# Emit one line per significant pipeline event. Covers failure signatures deliberately:
# a filter that matched only success would stay silent through a crashloop.
cd /home/patrickd/glm-5.3-reap
LAST=$(.venv/bin/python -c "
import sys;sys.path.insert(0,'scripts')
from common import db
with db() as c:
    r=c.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()
print(r[0])" 2>/dev/null || echo 0)
while true; do
  OUT=$(.venv/bin/python -c "
import sys;sys.path.insert(0,'scripts')
from common import db
last=$LAST
rows=[]
with db() as c:
    for i,ts,stg,lv,msg in c.execute(
        'SELECT id,ts,stage,level,msg FROM events WHERE id>? ORDER BY id', (last,)):
        m=msg.replace(chr(10),' ')
        if lv in ('ERROR','WARN') or m.startswith(('START','DONE','FAILED')) \
           or 'GATE' in m or 'PROJECTION' in m or 'pipeline start' in m \
           or 'all stages terminal' in m or 'published' in m:
            rows.append(f'{ts[11:19]} {lv} {stg}: {m[:150]}')
        last=i
print(chr(10).join(rows))
print('__LAST__'+str(last))
" 2>/dev/null)
  NEW=$(echo "$OUT" | grep '^__LAST__' | sed 's/__LAST__//')
  echo "$OUT" | grep -v '^__LAST__' | grep -v '^$'
  [ -n "$NEW" ] && LAST=$NEW
  sleep 90
done
