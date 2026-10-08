import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

PATH = "/Users/ruchitazingade/Desktop/MSE coursework/semester III/Applied Macro and Financial Econometrics/pricing_strategy/"

transactions = pd.read_csv(PATH + 'transaction_data.csv')
products = pd.read_csv(PATH + 'product.csv')
campaigns = pd.read_csv(PATH + 'campaign_table.csv')

print("Transactions shape:", transactions.shape)
print("Products shape:", products.shape)
print(transactions.head())
print(transactions.columns.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: MERGE AND CLEAN
# ─────────────────────────────────────────────────────────────────────────────

df = transactions.merge(
    products[['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC']],
    on='PRODUCT_ID', how='left'
)

# Week number
df['WEEK'] = df['DAY'] // 7 + 1

# Unit price
df['UNIT_PRICE'] = df['SALES_VALUE'] / (df['QUANTITY'] + 1e-9)

# Remove zero/negative prices and outliers
df = df[df['UNIT_PRICE'] > 0]
df = df[df['UNIT_PRICE'] < df['UNIT_PRICE'].quantile(0.99)]
df = df[df['QUANTITY'] > 0]

print("\nAfter cleaning:", df.shape)
print("Departments:", df['DEPARTMENT'].nunique())
print("Products:", df['PRODUCT_ID'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: AGGREGATE TO PRODUCT-WEEK LEVEL
# ─────────────────────────────────────────────────────────────────────────────

weekly = df.groupby(
    ['PRODUCT_ID', 'COMMODITY_DESC', 'DEPARTMENT', 'WEEK']
).agg(
    Total_Quantity=('QUANTITY', 'sum'),
    Avg_Price=('UNIT_PRICE', 'mean'),
    Total_Revenue=('SALES_VALUE', 'sum'),
    Transactions=('household_key', 'nunique')
).reset_index()

# Keep products with at least 10 weeks of data
product_weeks = weekly.groupby('PRODUCT_ID')['WEEK'].count()
valid_products = product_weeks[product_weeks >= 10].index
weekly = weekly[weekly['PRODUCT_ID'].isin(valid_products)]

print("\nWeekly dataset shape:", weekly.shape)
print("Valid products:", weekly['PRODUCT_ID'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: PRICE ELASTICITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_elasticity(product_df):
    if len(product_df) < 8:
        return None
    
    log_price = np.log(product_df['Avg_Price'] + 1e-9)
    log_qty = np.log(product_df['Total_Quantity'] + 1e-9)
    
    # Fix: add constant separately, don't rename columns
    X = sm.add_constant(log_price.values)
    
    try:
        model = sm.OLS(log_qty.values, X).fit()
        return {
            'elasticity': model.params[1],  # index instead of name
            'r_squared': model.rsquared,
            'p_value': model.pvalues[1],
            'n_weeks': len(product_df),
            'avg_price': product_df['Avg_Price'].mean(),
            'avg_quantity': product_df['Total_Quantity'].mean(),
            'avg_revenue': product_df['Total_Revenue'].mean()
        }
    except:
        return None
    
elasticities = []
for product_id, group in weekly.groupby('PRODUCT_ID'):
    
    # NEW: skip products with very low price variation
    price_cv = group['Avg_Price'].std() / (group['Avg_Price'].mean() + 1e-9)
    if price_cv < 0.01:  # less than 1% coefficient of variation
        continue
    
    result = calculate_elasticity(group)
    if result:
        result['PRODUCT_ID'] = product_id
        elasticities.append(result)

elasticity_df = pd.DataFrame(elasticities)

# Keep statistically significant results
elasticity_df = elasticity_df[elasticity_df['p_value'] < 0.05]

# NEW: remove extreme elasticity values — economically unrealistic
elasticity_df = elasticity_df[elasticity_df['elasticity'].between(-10, 5)]

# Merge product info
elasticity_df = elasticity_df.merge(
    products[['PRODUCT_ID', 'COMMODITY_DESC', 'DEPARTMENT']],
    on='PRODUCT_ID', how='left'
)

print("\nElasticity estimates:", len(elasticity_df))
print("Elasticity range:", elasticity_df['elasticity'].min().round(2),
      "to", elasticity_df['elasticity'].max().round(2))
print("\nSample output:")
print(elasticity_df[['COMMODITY_DESC', 'DEPARTMENT', 'elasticity',
                       'avg_price', 'avg_revenue']].head(10))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SEGMENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_elasticity(e):
    if e > -0.5:
        return 'Inelastic'
    elif e > -1.0:
        return 'Slightly Elastic'
    elif e > -1.5:
        return 'Elastic'
    else:
        return 'Highly Elastic'

def pricing_recommendation(e):
    if e > -0.5:
        return 'INCREASE PRICE'
    elif e > -1.0:
        return 'HOLD PRICE'
    elif e > -1.5:
        return 'USE PROMOTIONS'
    else:
        return 'REDUCE PRICE'

elasticity_df['Segment'] = elasticity_df['elasticity'].apply(classify_elasticity)
elasticity_df['Recommendation'] = elasticity_df['elasticity'].apply(pricing_recommendation)

print("\nSegment distribution:")
print(elasticity_df['Segment'].value_counts())

# Save for dashboard use tomorrow
elasticity_df.to_csv(PATH + 'elasticity_output.csv', index=False)
weekly.to_csv(PATH + 'weekly_output.csv', index=False)
print("\nFiles saved. Day 1 complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: REVENUE OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

from scipy.optimize import minimize_scalar

def optimize_price(current_price, current_qty, elasticity,
                   price_floor=0.5, price_ceiling=2.0):
    def negative_revenue(price_multiplier):
        new_price = current_price * price_multiplier
        new_qty = current_qty * (price_multiplier ** elasticity)
        return -(new_price * new_qty)

    result = minimize_scalar(
        negative_revenue,
        bounds=(price_floor, price_ceiling),
        method='bounded'
    )

    optimal_multiplier = result.x
    optimal_price = current_price * optimal_multiplier
    optimal_qty = current_qty * (optimal_multiplier ** elasticity)
    optimal_revenue = optimal_price * optimal_qty
    current_revenue = current_price * current_qty
    revenue_uplift = (optimal_revenue - current_revenue) / current_revenue * 100

    return {
        'optimal_price': round(optimal_price, 4),
        'price_change_pct': round((optimal_multiplier - 1) * 100, 2),
        'optimal_revenue': round(optimal_revenue, 4),
        'revenue_uplift_pct': round(revenue_uplift, 2)
    }

print("\nRunning revenue optimization...")
optimization_results = []
for _, row in elasticity_df.iterrows():
    opt = optimize_price(
        row['avg_price'],
        row['avg_quantity'],
        row['elasticity']
    )
    opt['PRODUCT_ID'] = row['PRODUCT_ID']
    optimization_results.append(opt)

opt_df = pd.DataFrame(optimization_results)
elasticity_df = elasticity_df.merge(opt_df, on='PRODUCT_ID', how='left')

# Remove positive elasticities — economically not meaningful for pricing
elasticity_df = elasticity_df[elasticity_df['elasticity'] < 0]

# Re-run optimization on clean dataset only
optimization_results = []
for _, row in elasticity_df.iterrows():
    opt = optimize_price(
        row['avg_price'],
        row['avg_quantity'],
        row['elasticity']
    )
    opt['PRODUCT_ID'] = row['PRODUCT_ID']
    optimization_results.append(opt)

opt_df = pd.DataFrame(optimization_results)
elasticity_df = elasticity_df.drop(
    columns=['optimal_price', 'price_change_pct', 
             'optimal_revenue', 'revenue_uplift_pct'],
    errors='ignore'
)
elasticity_df = elasticity_df.merge(opt_df, on='PRODUCT_ID', how='left')

# Cap uplift at 100% — anything above is optimizer artifact
elasticity_df['revenue_uplift_pct'] = elasticity_df['revenue_uplift_pct'].clip(upper=100)

# Sanity check
print(f"Revenue uplift range: {elasticity_df['revenue_uplift_pct'].min():.1f}% to {elasticity_df['revenue_uplift_pct'].max():.1f}%")
print(f"Average revenue uplift: {elasticity_df['revenue_uplift_pct'].mean():.1f}%")
print("\nSample optimization output:")
print(elasticity_df[['COMMODITY_DESC', 'avg_price', 'optimal_price', 
                       'price_change_pct', 'revenue_uplift_pct']].head(10))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: DEPARTMENT SUMMARY + STRATEGIC PRIORITY
# ─────────────────────────────────────────────────────────────────────────────

dept_summary = elasticity_df.groupby('DEPARTMENT').agg(
    Avg_Elasticity=('elasticity', 'mean'),
    Products_Analyzed=('PRODUCT_ID', 'count'),
    Avg_Revenue_Uplift=('revenue_uplift_pct', 'mean'),
    Total_Revenue=('avg_revenue', 'sum'),
    Inelastic_Products=('Segment', lambda x: (x == 'Inelastic').sum()),
    Elastic_Products=('Segment', lambda x: (x == 'Highly Elastic').sum())
).reset_index()

dept_summary['Strategic_Priority'] = dept_summary.apply(
    lambda x: 'INVEST'
    if x['Avg_Elasticity'] > -1.0 and x['Avg_Revenue_Uplift'] > 10
    else 'OPTIMIZE'
    if x['Avg_Elasticity'] > -1.5
    else 'REVIEW',
    axis=1
)

print("\nDepartment Strategic Summary:")
print(dept_summary[['DEPARTMENT', 'Avg_Elasticity',
                     'Avg_Revenue_Uplift', 'Strategic_Priority']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL FILES
# ─────────────────────────────────────────────────────────────────────────────

elasticity_df.to_csv(PATH + 'elasticity_output.csv', index=False)
weekly.to_csv(PATH + 'weekly_output.csv', index=False)
dept_summary.to_csv(PATH + 'dept_summary.csv', index=False)

print("\nAll three files saved.")
print("elasticity_output.csv —", len(elasticity_df), "rows")
print("weekly_output.csv —", len(weekly), "rows")
print("dept_summary.csv —", len(dept_summary), "rows")
print("\nDay 2 and 3 complete. Ready for dashboard.")