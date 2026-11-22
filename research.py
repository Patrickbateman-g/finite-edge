"""Chronological forecasting and exploratory diagnostics; standard library only.
Completed-session features predict next-session open-to-close return.
"""
import csv
import json
import math
import statistics as st
import hashlib
import random
from pathlib import Path
ROOT = Path(__file__).parent
FEATURES = ['return_1d', 'momentum_5d', 'momentum_20d', 'volatility_20d', 'range_1d', 'relative_volume']
TRAIN_END, TEST_START, REFIT_EVERY = 240, 320, 20
POSITION_SCALE, ONE_WAY_BPS = .01, 3.5
PENALTIES = [.001, .01, .1, 1, 10, 100]


def solve(a, b):
    """Small dense system with partial-pivot Gauss-Jordan elimination."""
    a = [list(row) + [y] for row, y in zip(a, b)]
    n = len(b)
    for i in range(n):
        k = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[k] = a[k], a[i]
        if abs(a[i][i]) < 1e-14:
            raise ValueError('singular system')
        pivot = a[i][i]
        a[i] = [v / pivot for v in a[i]]
        for j in range(n):
            if i != j:
                multiplier = a[j][i]
                a[j] = [v - multiplier * w for v, w in zip(a[j], a[i])]
    return [r[-1] for r in a]


def fit(rows, penalty):
    """Mean squared error + penalty * L2; intercept is unpenalized.
    Centering/scaling are learned from supplied training rows only.
    """
    if not rows or penalty < 0 or not math.isfinite(penalty):
        raise ValueError('nonempty rows and finite nonnegative penalty required')
    xs, ys = [r['x'] for r in rows], [r['y'] for r in rows]
    means = [st.mean(c) for c in zip(*xs)]
    scales = [st.pstdev(c) or 1 for c in zip(*xs)]
    z = [[1] + [(v-m)/s for v,m,s in zip(x,means,scales)] for x in xs]
    n = len(z[0])
    a = [[sum(r[i]*r[j] for r in z)/len(z) + (penalty if i == j and i else 0)
          for j in range(n)] for i in range(n)]
    b = [sum(r[i]*y for r,y in zip(z,ys))/len(z) for i in range(n)]
    return dict(means=means, scales=scales, beta=solve(a,b))


def predict(model, x):
    return model['beta'][0] + sum(b*(v-m)/s for b,v,m,s in
        zip(model['beta'][1:],x,model['means'],model['scales']))


def load_rows():
    with (ROOT/'data/apple.csv').open() as f:
        raw = list(csv.DictReader(f))
    rows = []
    for i in range(21,len(raw)-1):
        r = raw[i]
        close = float(r['AAPL.Adjusted'])
        trailing = raw[i-19:i+1]
        returns = [float(raw[j]['AAPL.Adjusted'])/float(raw[j-1]['AAPL.Adjusted'])-1
                   for j in range(i-19,i+1)]
        x = [returns[-1], close/float(raw[i-5]['AAPL.Adjusted'])-1,
             close/float(raw[i-20]['AAPL.Adjusted'])-1, st.pstdev(returns),
             (float(r['AAPL.High'])-float(r['AAPL.Low']))/float(r['AAPL.Close']),
             float(r['AAPL.Volume'])/st.mean(float(t['AAPL.Volume']) for t in trailing)-1]
        nxt = raw[i+1]
        rows.append(dict(date=nxt['Date'],feature_date=r['Date'],x=x,
            y=float(nxt['AAPL.Close'])/float(nxt['AAPL.Open'])-1,
            adv=st.mean(float(t['AAPL.Close'])*float(t['AAPL.Volume']) for t in trailing)))
    return rows


def validation_curve(train, valid):
    scores = []
    for p in PENALTIES:
        model = fit(train,p)
        scores.append(dict(penalty=p,
            train_mse=st.mean((predict(model,r['x'])-r['y'])**2 for r in train),
            mse=st.mean((predict(model,r['x'])-r['y'])**2 for r in valid),
            coefficient_l2=math.sqrt(sum(b*b for b in model['beta'][1:])),
            coefficients=model['beta'][1:]))
    return scores


def trading_row(row, forecast):
    w = max(-1,min(1,forecast/POSITION_SCALE))
    turnover = 2*abs(w)  # Open entry and close exit, each day starts flat.
    cost = turnover*ONE_WAY_BPS/10000
    gross = w*row['y']
    return dict(date=row['date'],feature_date=row['feature_date'],prediction=forecast,
        actual=row['y'],weight=w,gross=gross,turnover=turnover,fixed_cost=cost,
        net=gross-cost,vol=row['x'][3],adv=row['adv'])


def performance(series):
    mse = st.mean((r['prediction']-r['actual'])**2 for r in series)
    base = st.mean(r['actual']**2 for r in series)
    net = [r['net'] for r in series]
    sd = st.pstdev(net)
    wealth = peak = 1.0
    drawdown = 0.0
    for r in series:
        wealth *= 1+r['net']
        peak = max(peak,wealth)
        drawdown = max(drawdown,1-wealth/peak)
    return dict(mse=mse,zero_mse=base,oos_r2=1-mse/base if base else None,
        annual_gross=252*st.mean(r['gross'] for r in series),
        annual_fixed_cost=252*st.mean(r['fixed_cost'] for r in series),
        annual_net_before_impact=252*st.mean(net),
        daily_turnover=st.mean(r['turnover'] for r in series),
        annual_volatility=math.sqrt(252)*sd,
        sharpe=math.sqrt(252)*st.mean(net)/sd if sd else None,
        max_drawdown=drawdown,cumulative_net=wealth-1,
        daily_vol=st.mean(r['vol'] for r in series),adv=st.mean(r['adv'] for r in series))


def walk_forward(rows, penalty, kind='ridge', drop_feature=None, start=TEST_START,
                 refit_every=REFIT_EVERY):
    """Batches use only earlier realized labels, available before next open.
    The one-session targets do not overlap across the training/test boundary.
    """
    if kind not in {'ridge','zero','expanding_mean','momentum_5d'}:
        raise ValueError('unknown baseline')
    selected = [i for i in range(len(FEATURES)) if i != drop_feature]
    out, folds = [], []
    for begin in range(start,len(rows),refit_every):
        train, batch = rows[:begin], rows[begin:begin+refit_every]
        if kind == 'ridge':
            subset = [dict(r,x=[r['x'][i] for i in selected]) for r in train]
            model = fit(subset,penalty)
        train_mean = st.mean(r['y'] for r in train)
        fold_series = []
        for r in batch:
            if kind == 'ridge':
                p = predict(model,[r['x'][i] for i in selected])
            elif kind == 'expanding_mean':
                p = train_mean
            elif kind == 'momentum_5d':
                p = r['x'][1]/5
            else:
                p = 0.0
            fold_series.append(trading_row(r,p))
        folds.append(dict(train_end=train[-1]['date'],test_start=batch[0]['date'],
            test_end=batch[-1]['date'],n_train=begin,n_test=len(batch),
            metrics=performance(fold_series)))
        out.extend(fold_series)
    gross_wealth = net_wealth = 1.0
    for r in out:
        gross_wealth *= 1+r['gross']
        net_wealth *= 1+r['net']
        r['cumulative_gross'],r['cumulative_net'] = gross_wealth-1,net_wealth-1
    return out,folds


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    low = int(index)
    high = min(low+1,len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def bootstrap(series, block_length=5, replicates=2000, seed=1729):
    """Circular moving-block percentile intervals conditional on fitted forecasts.
    Paired daily observations share resampling indices; no refitting within draws.
    """
    if not series or block_length < 1 or replicates < 2:
        raise ValueError('nonempty series, positive block length, at least two draws required')
    rng = random.Random(seed)
    n = len(series)
    triples = [(r['net'],(r['prediction']-r['actual'])**2,r['actual']**2) for r in series]
    annual_net,r_squared = [],[]
    for _ in range(replicates):
        sample = []
        while len(sample) < n:
            begin = rng.randrange(n)
            sample.extend(triples[(begin+i)%n] for i in range(block_length))
        sample = sample[:n]
        annual_net.append(252*sum(t[0] for t in sample)/n)
        denominator = sum(t[2] for t in sample)
        r_squared.append(1-sum(t[1] for t in sample)/denominator if denominator else 0)
    interval = lambda values: [quantile(values,.025),quantile(values,.975)]
    return dict(block_length=block_length,replicates=replicates,seed=seed,
        annual_net_ci95=interval(annual_net),oos_r2_ci95=interval(r_squared))


def build_result():
    rows = load_rows()
    train,valid,test = rows[:TRAIN_END],rows[TRAIN_END:TEST_START],rows[TEST_START:]
    scores = validation_curve(train,valid)
    penalty = min(scores,key=lambda r:r['mse'])['penalty']
    initial = fit(rows[:TEST_START],penalty)
    out,folds = walk_forward(rows,penalty)
    metrics = performance(out)
    comparisons = []
    definitions = [
        ('zero','Zero forecast','Predict zero; stay in cash.'),
        ('expanding_mean','Expanding mean','Mean of observed training targets; refit every 20 sessions.'),
        ('momentum_5d','5-day momentum','Previous-close 5-day adjusted return divided by five; an untuned heuristic.'),
        ('ridge','Ridge regression','Six standardized features; validation-selected L2 penalty.')]
    for kind,label,definition in definitions:
        path = out if kind == 'ridge' else walk_forward(rows,penalty,kind)[0]
        comparisons.append(dict(id=kind,label=label,definition=definition,metrics=performance(path),
            equity=[dict(date=r['date'],cumulative_net=r['cumulative_net']) for r in path]))
    ablations = []
    for i,feature in enumerate(FEATURES):
        path,_ = walk_forward(rows,penalty,drop_feature=i)
        score = performance(path)
        ablations.append(dict(feature=feature,label='Without '+feature,mse=score['mse'],
            oos_r2=score['oos_r2'],delta_mse=score['mse']-metrics['mse'],
            annual_net_before_impact=score['annual_net_before_impact']))
    uncertainty = bootstrap(out)
    uncertainty.update(method='Circular moving-block percentile bootstrap of paired daily observations',
        confidence=.95,block_sensitivity=[bootstrap(out,block_length=k) for k in [1,5,10]],
        caveat='Conditional on fitted forecasts. No refitting inside bootstrap; excludes model-selection '
        'uncertainty and future regime changes. Post-hoc diagnostics on a previously inspected short holdout.')
    return dict(schema_version=2,title='AAPL · historical walk-forward experiment',
        source='https://github.com/plotly/datasets/blob/master/finance-charts-apple.csv',
        sha256=hashlib.sha256((ROOT/'data/apple.csv').read_bytes()).hexdigest(),
        features=FEATURES,n=len(rows),raw_n=len(rows)+22,
        train_n=len(train),validation_n=len(valid),test_n=len(test),
        train_start=train[0]['date'],train_end=train[-1]['date'],
        validation_start=valid[0]['date'],validation_end=valid[-1]['date'],
        test_start=test[0]['date'],test_end=test[-1]['date'],selected_penalty=penalty,
        validation_scores=scores,folds=folds,model=initial,metrics=metrics,series=out,
        comparison=comparisons,ablations=ablations,uncertainty=uncertainty,
        cost_sensitivity=[dict(one_way_bps=bps,annual_net=metrics['annual_gross']-
            252*metrics['daily_turnover']*bps/10000) for bps in [0,1,3.5,7,15]],
        protocol=dict(target='Following-session close / open − 1',
            information_set='All features available at previous session close',
            selection='Select smallest chronological validation MSE; freeze penalty for test.',
            retraining='Expand training window every 20 sessions; only earlier realized labels.',
            positions='Clip forecast / 0.01 to [−1,+1]; open entry, close exit; identical across baselines.',
            costs='5 bps full spread / 2 + 1 bp fee = 3.5 bps per side; both legs charged.',
            ablation='Drop each feature; keep full-model selected penalty and refit schedule fixed.',
            r_squared='1 − model squared error / zero-forecast squared error; not centered textbook R².',
            performance='Annual arithmetic return ×252; volatility and Sharpe use √252 without '
                'autocorrelation adjustment. Drawdown and equity compound daily net returns.',
            research_status='Exploratory retrospective case study. Diagnostic menu expanded after '
                'holdout inspection; not an untouched confirmatory test.'),
        limitations=[
            'Single surviving stock, 506 raw days and 164 evaluation sessions; not a current market estimate.',
            'Selection uses chronological validation only; new diagnostics are post-hoc, not preregistered.',
            'Prior-close features, following open entry and close exit; executions and costs are approximations.',
            'No borrow fees, auction depth, observed spreads, taxes, or true execution data.',
            'Bootstrap intervals condition on fitted forecasts and this historical regime.',
            'Short-sample annualized returns and unadjusted Sharpe are descriptive; no persistent edge established.',
            'Capacity scenarios are hypothetical; negative ridge net return supports zero deployable capital for that model.'])


def main():
    result = build_result()
    (ROOT/'dist/research.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'dist/research-data.js').write_text('window.RESEARCH = '+json.dumps(result)+';\n')
    print(json.dumps(dict(metrics=result['metrics'],uncertainty=result['uncertainty'],
        comparison=[dict(id=r['id'],metrics=r['metrics']) for r in result['comparison']]),indent=2))
    print('Test sessions',result['test_n'])


if __name__ == '__main__':
    main()
