import json
with open('debug-lora-nan-loss-step2.ndjson') as f:
    events = [json.loads(line) for line in f]
opt_states = [e for e in events if e.get('event') == 'opt_state']
print(f'Number of opt_state events: {len(opt_states)}')
for i, e in enumerate(opt_states):
    if i < 3:
        s = e['step']
        keys = e.get('state_keys', [])
        sq = e.get('exp_avg_sq_dtype', None)
        sq_nan = e.get('exp_avg_sq_has_nan', None)
        avg_nan = e.get('exp_avg_has_nan', None)
        print(f"  step {s}: keys={keys}, exp_avg_sq_dtype={sq}, sq_nan={sq_nan}, avg_nan={avg_nan}")
params = [e for e in events if e.get('event') == 'param_after_step']
for e in params:
    s = e['step']
    nan = e['first_param_nan']
    mx = e['first_param_max_abs']
    print(f"  step {s}: nan={nan}, max={mx}")
