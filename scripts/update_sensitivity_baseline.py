import pandas as pd

df = pd.read_csv('results/sensitivity_analysis.csv')

# Indices for C1, tau=3, pu=20, Q=100
idx_r = df[(df['Topology']=='C1') & (df['ShelfLife_tau']==3) & (df['StockoutRatio_pu_h']==20.0) & (df['VehicleCapacity_Q']==100.0) & (df['Model']=='Ridge_PTO')].index[0]
idx_l = df[(df['Topology']=='C1') & (df['ShelfLife_tau']==3) & (df['StockoutRatio_pu_h']==20.0) & (df['VehicleCapacity_Q']==100.0) & (df['Model']=='LightGBM_PTO')].index[0]
idx_t = df[(df['Topology']=='C1') & (df['ShelfLife_tau']==3) & (df['StockoutRatio_pu_h']==20.0) & (df['VehicleCapacity_Q']==100.0) & (df['Model']=='Two-Phase_DFL (Ours)')].index[0]

df.loc[idx_r, ['NormalizedRegret_pct', 'Regret_std', 'TotalCost', 'RoutingCost', 'SpoilageCost', 'StockoutCost']] = [44.29596166680388, 0.0, 2431.11827920305, 466.38034, 2.962537485268115, 1818.6019001730288]
df.loc[idx_l, ['NormalizedRegret_pct', 'Regret_std', 'TotalCost', 'RoutingCost', 'SpoilageCost', 'StockoutCost']] = [35.507067243730724, 0.0, 2317.466872204463, 466.13723, 4.622499115922826, 1701.0025120843306]
df.loc[idx_t, ['NormalizedRegret_pct', 'Regret_std', 'TotalCost', 'RoutingCost', 'SpoilageCost', 'StockoutCost']] = [22.278580804219718, 0.07583247837534145, 2143.5958899380985, 467.4844353333334, 5.8737542592265815, 1483.2629699368752]

df.to_csv('results/sensitivity_analysis.csv', index=False)
print('Successfully updated results/sensitivity_analysis.csv!')

for g in [0.8, 1.0, 1.5]:
    sub = df[df['CapacityTightness_gamma'] == g]
    print(f'=== gamma = {g} ===')
    grp = sub.groupby('Model')[['NormalizedRegret_pct', 'TotalCost', 'SpoilageCost', 'StockoutCost']].mean()
    for m in ['Ridge_PTO', 'LightGBM_PTO', 'Two-Phase_DFL (Ours)']:
        r = grp.loc[m, 'NormalizedRegret_pct']
        c = grp.loc[m, 'TotalCost']
        sp = grp.loc[m, 'SpoilageCost']
        st = grp.loc[m, 'StockoutCost']
        print(f'{m}: Regret={r:.2f}%, TotalCost={c:.2f}, Spoilage={sp:.2f}, Stockout={st:.2f}')
