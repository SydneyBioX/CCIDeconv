import joblib

scp_reg = joblib.load("supplementary_analysis/results/cascading_metrics/results_regression_scp.joblib")
print(f"ScP reg keys: {list(scp_reg.keys())}")
first_key = list(scp_reg.keys())[0]
print(f"First key: {first_key}, sub-keys: {list(scp_reg[first_key].keys())}")
for comp in ['cyt', 'nuc']:
    val = scp_reg[first_key][comp]
    print(f"  {comp} type: {type(val)}, len: {len(val)}")
    if len(val) > 1 and hasattr(val[1], 'head'):
        print(f"    cols: {list(val[1].columns)}, shape: {val[1].shape}")
        print(val[1].head(2))

tot_cyt = sum(len(scp_reg[k]['cyt'][1]) for k in scp_reg.keys())
tot_nuc = sum(len(scp_reg[k]['nuc'][1]) for k in scp_reg.keys())
print(f"Total ScP cyt predictions across all 9 datasets: {tot_cyt}")
print(f"Total ScP nuc predictions across all 9 datasets: {tot_nuc}")
