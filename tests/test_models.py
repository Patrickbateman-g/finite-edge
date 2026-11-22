import sys, unittest, math, copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import Scenario, capacity, evaluate
from research import load_rows, fit, predict, solve, walk_forward, trading_row, performance, bootstrap
class CapacityTests(unittest.TestCase):
    def test_zero_and_optimum(self):
        s=Scenario(participation=1);c=capacity(s)
        self.assertAlmostEqual(evaluate(s,c['break_even'])['net'],0,places=10)
        a=c['optimum'];self.assertGreater(evaluate(s,a)['profit'],evaluate(s,a*.99)['profit'])
        self.assertGreater(evaluate(s,a)['profit'],evaluate(s,a*1.01)['profit'])
    def test_impact_scaling(self):
        s=Scenario();self.assertAlmostEqual(evaluate(s,4e6)['impact'],2*evaluate(s,1e6)['impact'])
    def test_no_edge(self):self.assertEqual(capacity(Scenario(alpha=0))['usable'],0)
    def test_liquidity_limit(self):
        s=Scenario(participation=.001);c=capacity(s);self.assertAlmostEqual(evaluate(s,c['operational'])['participation'],.001)
    def test_invalid(self):
        for s in [Scenario(adv=0),Scenario(turnover=0),Scenario(names=1.5),Scenario(alpha=float('nan'))]:
            with self.assertRaises(ValueError):capacity(s)
    def test_delay_does_not_invent_liquidity(self):
        a,b=Scenario(days=1),Scenario(days=5)
        self.assertEqual(capacity(a)['operational'],capacity(b)['operational'])
        self.assertLess(capacity(b)['break_even'],capacity(a)['break_even'])
class MLTests(unittest.TestCase):
    def test_linear_solver(self):self.assertEqual(solve([[2,0],[0,4]],[6,8]),[3,2])
    def test_temporal_order(self):
        rows=load_rows();self.assertTrue(all(r['feature_date']<r['date'] for r in rows));self.assertEqual(len(set(r['date'] for r in rows)),len(rows))
    def test_future_rows_cannot_change_forecast_prefix(self):
        rows=load_rows();before,_=walk_forward(rows,.1)
        changed=copy.deepcopy(rows)
        for r in changed[360:]:
            r['y']+=.2
            r['x']=[x*7+.3 for x in r['x']]
        after,_=walk_forward(changed,.1)
        self.assertEqual(before[:40],after[:40])
        self.assertNotEqual(before[40]['prediction'],after[40]['prediction'])
    def test_current_target_only_enters_at_next_refit(self):
        rows=load_rows();before,_=walk_forward(rows,.1)
        changed=copy.deepcopy(rows);changed[320]['y']+=.2
        after,_=walk_forward(changed,.1)
        self.assertEqual([r['prediction'] for r in before[:20]],
                         [r['prediction'] for r in after[:20]])
        self.assertNotEqual(before[20]['prediction'],after[20]['prediction'])
    def test_dropped_feature_does_not_affect_ablation(self):
        rows=load_rows();before,_=walk_forward(rows,.1,drop_feature=0)
        changed=copy.deepcopy(rows)
        for i,r in enumerate(changed):r['x'][0]=i*100
        after,_=walk_forward(changed,.1,drop_feature=0)
        self.assertEqual(before,after)
    def test_ridge_matches_closed_form_shrinkage(self):
        rows=[{'x':[x],'y':3+2*x} for x in [-1,1,-1,1]]
        model=fit(rows,3)
        self.assertAlmostEqual(model['beta'][0],3)
        self.assertAlmostEqual(model['beta'][1],.5)
    def test_costs_charge_both_legs(self):
        r=trading_row(load_rows()[0],.005)
        self.assertEqual(r['weight'],.5)
        self.assertEqual(r['turnover'],1)
        self.assertAlmostEqual(r['fixed_cost'],.00035)
        self.assertAlmostEqual(r['net'],r['gross']-.00035)
    def test_bootstrap_is_paired_and_reproducible(self):
        series=[{'net':.001,'prediction':0,'actual':i*.01} for i in range(1,20)]
        a=bootstrap(series,block_length=5,replicates=100)
        self.assertEqual(a,bootstrap(series,block_length=5,replicates=100))
        self.assertEqual(a['oos_r2_ci95'],[0,0])
        for bound in a['annual_net_ci95']:self.assertAlmostEqual(bound,.252)
    def test_drawdown_uses_compounded_wealth(self):
        series=[dict(prediction=0,actual=x,net=x,gross=x,fixed_cost=0,
                     turnover=0,vol=.01,adv=1e6) for x in [-.1,.1]]
        result=performance(series)
        self.assertAlmostEqual(result['cumulative_net'],-.01)
        self.assertAlmostEqual(result['max_drawdown'],.1)
    def test_constant_target(self):
        rows=[{'x':[i,i*i],'y':.01} for i in range(20)];model=fit(rows,.1)
        self.assertAlmostEqual(predict(model,[12,144]),.01,places=10)
if __name__=='__main__':unittest.main()
