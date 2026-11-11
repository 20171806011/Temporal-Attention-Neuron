"""Post-run accounting correction only; frozen dynamics and outcomes stay unchanged."""
from pathlib import Path
import json,datetime
import numpy as np
import engine
import verify
R=Path(__file__).resolve().parent
rows=json.loads((R/'mechanism/ROWS.json').read_text(encoding='utf-8'))
specs={r['name']:r for r in json.loads((R/'mechanism/SPECS.json').read_text(encoding='utf-8'))}
corrected=[]
for row in rows:
    spec=specs.get(row['condition'],{})
    if row['arm']!='POINT' or spec.get('kind')!='path' or spec.get('path')!='local':continue
    s=row['seed'];data=np.load(R/'mechanism'/f'data_s{s}.npz');w=np.load(R/'weights'/f's{s}_POINT_{row["cohort"]}.npy')
    a=engine.forward(data['x'],w,'POINT',spec=spec);b=verify.native(data['x'],w,'POINT',spec=spec)
    assert np.array_equal(a['spikes'],b['spikes'])
    emitted=a['spikes'][:,:,:12].sum((1,2))
    blocked=a['spikes'][:,spec['window'],:12].sum((1,2))
    delivered=emitted-blocked
    check_from_loop=np.zeros(len(data['y']))
    for tick in range(24):
        if tick not in spec['window']:check_from_loop+=b['spikes'][:,tick,:12].sum(1)
    assert np.array_equal(delivered,check_from_loop)
    old=np.load(R/'mechanism'/f'{row["id"]}.npz')['counts'];assert np.array_equal(a['counts'],old)
    corrected.append(dict(id=row['id'],blocked_branch_events_mean=float(blocked.mean()),
        original_branch_event_deliveries=row['branch_event_deliveries'],corrected_branch_event_deliveries=float(delivered.mean()),
        original_consumed_internal_event_deliveries=row['consumed_internal_event_deliveries'],
        corrected_consumed_internal_event_deliveries=float(row['recurrent_event_deliveries']+row['hidden_output_deliveries']+delivered.mean())))
record=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),affected_rows=len(corrected),corrections=corrected,
    scope='Only POINT branch-to-soma cut rows overcounted delivered branch events. Native emitted events, dynamics, predictions, primary effects, training, and all fair-comparison resource rows are unchanged. Two separate forwards agree on all spikes; corrected delivery totals exclude the two blocked ticks.')
(R/'RESOURCE_CORRECTION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('Resource correction checked:',len(corrected),'rows')
