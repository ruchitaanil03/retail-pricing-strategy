# Retail Pricing Strategy & Portfolio Intelligence Tool

**Tools:** Python · Statsmodels · Scipy · Streamlit · Plotly · Power BI  
**Data:** Dunnhumby "The Complete Journey" — 2.5M+ retail transactions  
**GitHub:** https://github.com/ruchitaanil03/retail-pricing-strategy

---

## Power BI Dashboard

![Power BI Dashboard](powerbi_dashboard.png)

---

## Objective

The core question this project answers is: how sensitive are retail consumers to price changes, and where can a retailer systematically make more money by adjusting prices?

Using 2.5 million real grocery transactions from Dunnhumby's "The Complete Journey" dataset, this project estimates product-level price elasticity of demand for 5,599 products, identifies the revenue-maximizing price for each product using bounded optimization, and frames the entire retail portfolio through a strategic asset evaluation lens — classifying departments into invest, optimize, or review categories based on pricing power and revenue potential.

---

## Dataset

Three tables from Dunnhumby "The Complete Journey" (available on Kaggle):

| File | Description | Key Columns |
|---|---|---|
| `transaction_data.csv` | 2.5M+ purchase events | household_key, PRODUCT_ID, DAY, QUANTITY, SALES_VALUE |
| `product.csv` | Product metadata | PRODUCT_ID, DEPARTMENT, COMMODITY_DESC |
| `campaign_table.csv` | Household-campaign mapping | household_key, CAMPAIGN |

After merging transactions with product metadata, the working dataset contained: `PRODUCT_ID`, `DEPARTMENT`, `COMMODITY_DESC`, `DAY`, `QUANTITY`, `SALES_VALUE`, and derived columns `UNIT_PRICE` and `WEEK`.

---

## Data Cleaning

**Unit price derivation:**

```python
df['UNIT_PRICE'] = df['SALES_VALUE'] / (df['QUANTITY'] + 1e-9)
```

A small epsilon is added to the denominator to avoid division by zero for edge-case zero-quantity rows.

**Three filters applied:**

```python
df = df[df['UNIT_PRICE'] > 0]                                 # remove zero/negative prices
df = df[df['UNIT_PRICE'] < df['UNIT_PRICE'].quantile(0.99)]   # remove top 1% price outliers
df = df[df['QUANTITY'] > 0]                                   # remove refunds and voids
```

Zero and negative prices are coupon artifacts or data entry errors. The top 1% price outlier filter removes bulk orders and miscoded entries that would distort the regression slope. Zero-quantity rows are refunds or voids with no demand signal.

**Missing data:** The Dunnhumby dataset has no nulls in core transaction columns. The merge on PRODUCT_ID is a left join, so any unmatched product receives a null DEPARTMENT and is dropped naturally during group-by aggregation.

---

## Aggregation to Product-Week Level

Raw transactions are at the household-purchase level, which is too granular for demand curve estimation. Price and quantity need to vary at a comparable level of aggregation.

```python
weekly = df.groupby(['PRODUCT_ID', 'COMMODITY_DESC', 'DEPARTMENT', 'WEEK']).agg(
    Total_Quantity=('QUANTITY', 'sum'),
    Avg_Price=('UNIT_PRICE', 'mean'),
    Total_Revenue=('SALES_VALUE', 'sum'),
    Transactions=('household_key', 'nunique')
).reset_index()
```

Week number is derived as:

```python
df['WEEK'] = df['DAY'] // 7 + 1
```

**Why weekly?** Weekly aggregation smooths daily noise while preserving genuine price variation from promotions, markdowns, and restocking cycles. Daily data is too noisy. Monthly aggregation loses the variation needed to identify elasticity.

**Minimum data filter:** Only products with at least 10 weeks of observations are retained. Fewer observations make the OLS regression unreliable.

```python
product_weeks = weekly.groupby('PRODUCT_ID')['WEEK'].count()
valid_products = product_weeks[product_weeks >= 10].index
weekly = weekly[weekly['PRODUCT_ID'].isin(valid_products)]
```

**Price variation filter:** Products where the coefficient of variation of price is below 1% are dropped. If a product never changed price, the regression slope is undefined — you cannot estimate how quantity responds to price when price never moved.

```python
price_cv = group['Avg_Price'].std() / (group['Avg_Price'].mean() + 1e-9)
if price_cv < 0.01:
    continue
```

---

## Price Elasticity Estimation

### The Model

Log-log OLS regression is run separately for each product at the product-week level:
log(Quantity_it) = α + β · log(Price_it) + ε_it


Where:
- `i` = product
- `t` = week
- `alpha` = intercept (log of baseline quantity when price equals 1)
- `beta` = price elasticity of demand — the key parameter being estimated
- `epsilon` = error term capturing all other demand drivers

### Why Log-Log?

In a log-log specification, the slope coefficient beta is directly interpretable as a constant elasticity:
β = (% change in Quantity) / (% change in Price)


This follows from differentiating `log(Q) = alpha + beta * log(P)`:
dQ/Q = β · dP/P


So if beta = -1.8, a 1% rise in price causes a 1.8% fall in quantity sold that week. The log-log form also handles the right-skewed distributions of both price and quantity, and linearizes the standard power-law demand curve `Q = A * P^beta`.

### OLS Implementation

```python
import statsmodels.api as sm

def calculate_elasticity(product_df):
    log_price = np.log(product_df['Avg_Price'] + 1e-9)
    log_qty = np.log(product_df['Total_Quantity'] + 1e-9)
    X = sm.add_constant(log_price.values)
    model = sm.OLS(log_qty.values, X).fit()
    return {
        'elasticity': model.params[1],
        'r_squared': model.rsquared,
        'p_value': model.pvalues[1],
        'n_weeks': len(product_df),
        'avg_price': product_df['Avg_Price'].mean(),
        'avg_quantity': product_df['Total_Quantity'].mean(),
        'avg_revenue': product_df['Total_Revenue'].mean()
    }
```

The constant is added manually so the intercept alpha is estimated. `params[1]` is the slope coefficient on `log(Price)` — that is the elasticity.

### Statistical Filters

**p-value < 0.05** — only statistically significant elasticities are retained. The p-value tests the null hypothesis H0: beta = 0, meaning price has no effect on quantity demanded. At p < 0.05, there is less than a 5% probability of observing a slope this large by chance if the true elasticity were zero. This filter retains 5,599 products from a larger initial set.

**Elasticity range (-10, +5)** — economically implausible estimates are removed. An elasticity of -50 would imply a 1% price rise causes a 50% quantity drop, which is not credible in grocery retail. Positive elasticities (Giffen goods) are also removed as they are essentially non-existent in mass-market grocery data.

**R-squared** is recorded but not used as a filter. It tells you how much of the weekly quantity variation is explained by price alone. Many products have low R-squared because other factors (promotions, seasonality, store events) also drive quantity, but the elasticity estimate can still be statistically valid and significant.

---

## Segment Classification

Products are classified into four segments based on the estimated beta:

| Beta Range | Segment | Economic Meaning | Recommendation |
|---|---|---|---|
| beta > -0.5 | Inelastic | Consumers barely respond to price changes | INCREASE PRICE |
| -1.0 < beta <= -0.5 | Slightly Elastic | Mild quantity response to price | HOLD PRICE |
| -1.5 < beta <= -1.0 | Elastic | Meaningful quantity response to price | USE PROMOTIONS |
| beta <= -1.5 | Highly Elastic | Strong quantity response to price | REDUCE PRICE |

The threshold beta = -1.0 is the unit elasticity point. Above -1 (closer to zero), raising price increases total revenue because the percentage gain in price exceeds the percentage loss in volume. Below -1, raising price destroys more revenue through volume loss than it recovers through the higher unit price.

```python
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
```

---

## Revenue Optimization

### Demand Function

Under the constant elasticity assumption derived from the log-log model, the demand function is:
**Q(P) = Q0 · (P / P0)^β**


Where P0 is the current average price and Q0 is the current average quantity.

### Revenue as a Function of Price
R(P) = P · Q(P)
     = P · Q0 · (P / P0)^β
     = Q0 · P0^(-β) · P^(1 + β)
     
### Finding the Optimal Price

Analytically, taking the derivative of R(P) with respect to P and setting it to zero gives the revenue-maximizing price multiplier:
m* = -1 / β

However this is unbounded and breaks at beta = -1. Instead, Scipy bounded optimization is used with the price constrained to plus or minus 50% of the current price:

```python
from scipy.optimize import minimize_scalar

def optimize_price(current_price, current_qty, elasticity, price_floor=0.5, price_ceiling=2.0):
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
    return optimal_price, revenue_uplift
```

**Revenue uplift formula:**
Uplift % = (Optimal Revenue - Current Revenue) / Current Revenue × 100

Revenue uplift is capped at 100% to remove optimizer artifacts — without the cap, very elastic products receive extreme recommendations like reducing price to near-zero to maximize volume, which is economically meaningless.

```python
elasticity_df['revenue_uplift_pct'] = elasticity_df['revenue_uplift_pct'].clip(upper=100)
```

---

## Department Summary and Strategic Priority

```python
dept_summary = elasticity_df.groupby('DEPARTMENT').agg(
    Avg_Elasticity=('elasticity', 'mean'),
    Products_Analyzed=('PRODUCT_ID', 'count'),
    Avg_Revenue_Uplift=('revenue_uplift_pct', 'mean'),
    Total_Revenue=('avg_revenue', 'sum'),
    Inelastic_Products=('Segment', lambda x: (x == 'Inelastic').sum()),
    Elastic_Products=('Segment', lambda x: (x == 'Highly Elastic').sum())
).reset_index()
```

Strategic priority is assigned as:

```python
dept_summary['Strategic_Priority'] = dept_summary.apply(
    lambda x: 'INVEST' if x['Avg_Elasticity'] > -1.0 and x['Avg_Revenue_Uplift'] > 10
    else 'OPTIMIZE' if x['Avg_Elasticity'] > -1.5
    else 'REVIEW', axis=1
)
```

| Priority | Condition | Meaning |
|---|---|---|
| INVEST | Avg elasticity > -1.0 and uplift > 10% | Moderate pricing power with room to grow. Prioritize here. |
| OPTIMIZE | Avg elasticity > -1.5 | Moderate elasticity. Extract efficiency without aggressive price moves. |
| REVIEW | Otherwise | Highly elastic, thin margins. Question whether the assortment and pricing strategy are right. |

---

## Results

### Key Numbers

| Metric | Value |
|---|---|
| Products analyzed | 5,599 |
| Average price elasticity | -1.90 |
| Inelastic products (beta > -0.5) | 71 |
| Average revenue uplift potential | 47.24% |
| Departments covered | 16 |

**Average elasticity of -1.90** means that on average across the portfolio, a 1% price increase causes a 1.9% quantity drop. The portfolio is moderately elastic overall — consumers respond meaningfully to grocery price changes.

**71 inelastic products** have strong pricing power. Raising prices here grows revenue with minimal volume loss. These are the portfolio's core defensible assets.

**Average revenue uplift of 47.24%** is the average gap between current prices and revenue-maximizing prices. It is high because most products are priced somewhat away from their demand curve optimum, and the bounded constraint of plus or minus 50% still allows meaningful movement.

### Department Summary

| Department | Avg Elasticity | Products | Avg Uplift % | Strategic Priority |
|---|---|---|---|---|
| COSMETICS | -0.69 | 7 | 43.2 | INVEST |
| SALAD BAR | -0.72 | 1 | 21.2 | INVEST |
| COUP/STR & MFG | -0.84 | 1 | 11.8 | INVEST |
| FLORAL | -1.31 | 1 | 24.3 | OPTIMIZE |
| SEAFOOD-PCKGD | -1.43 | 68 | 32.6 | OPTIMIZE |
| SEAFOOD | -1.54 | 2 | 45.5 | REVIEW |
| DELI | -1.56 | 62 | 47.7 | REVIEW |
| DRUG GM | -1.61 | 708 | 46.9 | REVIEW |
| SPIRITS | -1.72 | 2 | 66.2 | REVIEW |
| NUTRITION | -1.84 | 118 | 59.1 | REVIEW |
| PASTRY | -1.87 | 73 | 57.6 | REVIEW |
| PRODUCE | -1.86 | 186 | 55.1 | REVIEW |
| GROCERY | -1.96 | 3872 | 56.9 | REVIEW |
| MEAT | -2.00 | 99 | 62.9 | REVIEW |
| CHEF SHOPPE | -2.03 | 4 | 60.9 | REVIEW |
| MEAT-PCKGD | -2.03 | 395 | 64.0 | REVIEW |

---

## Power BI Dashboard

### KPI Cards

| Card | Value | What It Means |
|---|---|---|
| Average Price Elasticity | -1.90 | Portfolio is moderately elastic — consumers respond to price |
| Products Analyzed | 5,599 | Number of products with statistically significant elasticity estimates |
| Average Revenue Uplift | 55.82% | Average gap between current and revenue-maximizing prices |
| Inelastic Products | 71 | Products with pricing power — raising price grows revenue |

### Recommendation Mix — Stacked Bar by Department

Each bar represents one department, stacked by recommendation category. Colors: green = HOLD PRICE, blue = INCREASE PRICE, yellow = REDUCE PRICE, red = USE PROMOTIONS.

What it shows: the composition of pricing strategy recommendations within each department. GROCERY is dominated by yellow and red because it is highly elastic — consumers switch easily and volume is very price-sensitive. COSMETICS skews blue, reflecting its stronger pricing power and inelastic demand.

### Segment Distribution — Donut Chart

Shows Inelastic, Slightly Elastic, Elastic, and Highly Elastic as shares of the 5,599-product portfolio.

What it shows: the majority of products fall in the Elastic or Highly Elastic segments, confirming the portfolio is broadly price-sensitive. The small Inelastic slice represents premium or habitual-purchase products where consumers are less responsive to price changes.

### Average Elasticity by Department — Horizontal Bar Chart

Departments sorted from least elastic (closest to zero, furthest right) to most elastic (most negative, furthest left).

What it shows: which departments have defensible pricing power. COSMETICS, SALAD BAR, and COUP/STR sit furthest right — consumers in these categories do not easily switch away when prices rise. MEAT-PCKGD and CHEF SHOPPE sit furthest left — buyers are highly responsive to price and will reduce purchases or substitute if prices increase.

Note: the X-axis is scaled to the data range rather than starting at zero, so all bars appear similar in length. Setting the axis minimum to 0 in Power BI would make the differences between departments visually clearer.

### Top 10 Products by Revenue Uplift — Horizontal Bar

The 10 individual products with the largest gap between current revenue and revenue at the optimal price, colored by recommendation type.

What it shows: where the highest individual product-level pricing opportunities exist. Products appearing here with INCREASE PRICE or HOLD PRICE recommendations are currently underpriced relative to their estimated demand curve — raising price would increase revenue without proportionate volume loss.

### Department Summary Table

Sortable table showing all 16 departments with Products_Analyzed, Total_Revenue, Avg_Elasticity, Avg_Revenue_Uplift, and Strategic_Priority. Conditional formatting applied to the Avg_Revenue_Uplift column — green shading for high uplift, red for low. Total row at the bottom aggregates all departments.

### Category Attractiveness Matrix — Scatter Plot

Each bubble is one product. The matrix maps the entire portfolio across two dimensions simultaneously.

| Axis | Variable | Interpretation |
|---|---|---|
| X-axis | avg_revenue | Market size and commercial importance of the product |
| Y-axis | avg elasticity | Pricing power — closer to 0 means stronger, more negative means weaker |
| Bubble size | revenue_uplift_pct | Size of the revenue optimization opportunity |
| Color | DEPARTMENT | Department grouping |

**The four quadrants:**

| Quadrant | Position | Label | Strategic Action |
|---|---|---|---|
| Top Right | High revenue + strong pricing power (elasticity closer to 0) | Core Assets | Protect. These are the most valuable products in the portfolio. |
| Top Left | Low revenue + strong pricing power | Growth | Invest. Pricing power exists but revenue is underdeveloped. |
| Bottom Right | High revenue + weak pricing power (highly elastic) | Mature | Optimize. Large products but price-sensitive — extract efficiency carefully. |
| Bottom Left | Low revenue + weak pricing power | Weak | Review. Small, elastic products with limited strategic value. |

GROCERY forms the largest cluster in the bottom-middle — enormous revenue but highly elastic demand. COSMETICS dots sit higher on the Y-axis reflecting stronger pricing power. MEAT-PCKGD has several large bubbles in the bottom-right quadrant — a large, commercially important department but highly price-sensitive, with significant revenue uplift opportunity if prices are carefully repositioned.

---

## Statistical Rigour Notes

All elasticity estimates retained in this analysis are statistically significant at p < 0.05 (two-tailed t-test on the log-log OLS slope coefficient).

R-squared values are recorded per product but not used as a filter. In grocery demand estimation, R-squared is typically low (0.15 to 0.35) because price is only one of many demand drivers — promotions, seasonality, and store events also matter. A low R-squared does not invalidate the elasticity estimate as long as the coefficient is significant.

The n_weeks column records how many weekly observations each product's regression was estimated on. Estimates based on more weeks of data are generally more stable. Products with fewer than 10 weeks were excluded at the aggregation stage.

Confidence intervals are available via `model.conf_int()` in statsmodels and could be added to a future version of the dashboard to show uncertainty bands around department-level elasticity estimates.

---

## Limitations

- Constant elasticity assumption from the log-log OLS model approximates the true demand curve. In reality, elasticity likely varies across the price range.
- Elasticity estimates are cross-sectional averages and do not account for seasonality or promotional effects.
- Revenue optimization ignores cross-price effects between substitute products. Raising the price of one product may shift demand to a close substitute in the same department.
- The price constraint of plus or minus 50% is operationally motivated but arbitrary. In practice, category managers would apply tighter constraints based on competitive positioning.

---

## Files

| File | Description |
|---|---|
| `pricing_project.py` | Full data pipeline — cleaning, elasticity estimation, optimization, export |
| `dashboard.py` | Streamlit 5-tab interactive dashboard |
| `retail_pricing_dashboard.pbix` | Power BI dashboard file |
| `retailpricingdashboard.pdf` | Power BI dashboard export |
| `elasticity_output.csv` | 5,599 products with elasticity, segments, optimal prices, revenue uplift |
| `dept_summary.csv` | Department-level strategic summary across 16 departments |
| `campaign_table.csv` | Dunnhumby campaign-household mapping |

---

## How to Run

```bash
pip install streamlit plotly statsmodels scipy pandas numpy

streamlit run dashboard.py
```

---

## Data Source

Dunnhumby "The Complete Journey"  
Available on Kaggle: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey  
Files used: `transaction_data.csv`, `product.csv`, `campaign_table.csv`
