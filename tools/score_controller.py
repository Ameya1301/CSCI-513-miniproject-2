import sys, json, glob, os
os.environ['EXACT'] = sys.argv[3]
sys.argv = ['x', sys.argv[1], sys.argv[2]]
src = open(sys.argv[1] + '/../repo/tools/offline_controller_check.py' if False else os.environ['H']).read()
sys.argv = ['x', sys.argv[1]]
exec(src[:src.index('fails = 0')])
noise, drop = (0.0, 0.0) if EXACT else (0.06, 0.03)
tot = {}; n = 0
for f in sorted(glob.glob(sys.argv[1] + '/test_data/*.json')):
    scn = json.load(open(f))
    for seed in range(10):
        res, _ = check(run(scn, seed, noise, drop)); n += 1
        for k, v in res.items(): tot[k] = tot.get(k, 0) + v
print('  '.join(f'{k}={tot[k]/n:.2f}' for k in tot), f' total={sum(tot.values())/n*20:.0f}%')
