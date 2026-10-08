import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Retail Pricing & Portfolio Strategy",
    layout="wide",
    page_icon="📊"
)

PATH = "/Users/ruchitazingade/Desktop/MSE coursework/semester III/Applied Macro and Financial Econometrics/pricing_strategy/"

@st.cache_data
def load_data():
    elasticity_df = pd.read_csv(PATH + 'elasticity_output.csv')
    weekly = pd.read_csv(PATH + 'weekly_output.csv')
    dept_summary = pd.read_csv(PATH + 'dept_summary.csv')
    return elasticity_df, weekly, dept_summary

elasticity_df, weekly, dept_summary = load_data()

st.sidebar.title("🔧 Filters")
st.sidebar.markdown("---")
dept_filter = st.sidebar.multiselect(
    "Select Department",
    options=sorted(elasticity_df['DEPARTMENT'].dropna().unique()),
    default=sorted(elasticity_df['DEPARTMENT'].dropna().unique())
)

filtered_df = elasticity_df[elasticity_df['DEPARTMENT'].isin(dept_filter)]
filtered_dept = dept_summary[dept_summary['DEPARTMENT'].isin(dept_filter)]

st.title("📊 Retail Pricing Strategy & Portfolio Intelligence Tool")
st.markdown("*Price elasticity analysis, revenue optimization, and strategic asset assessment — Dunnhumby Consumer Transaction Data*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Products Analyzed", f"{len(filtered_df):,}")
col2.metric("Avg Price Elasticity", f"{filtered_df['elasticity'].mean():.2f}")
col3.metric("Avg Revenue Uplift Potential", f"{filtered_df['revenue_uplift_pct'].mean():.1f}%")
col4.metric("Inelastic Products (Pricing Power)", f"{len(filtered_df[filtered_df['elasticity'] > -0.5]):,}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Portfolio Overview",
    "🔍 Product Explorer",
    "💰 Pricing Opportunities",
    "📈 Demand Curves",
    "🏛️ Strategic Portfolio"
])

# TAB 1
with tab1:
    st.subheader("Portfolio Overview")
    st.markdown("How price-sensitive are products across different departments?")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            filtered_dept.sort_values('Avg_Elasticity'),
            x='Avg_Elasticity',
            y='DEPARTMENT',
            orientation='h',
            color='Avg_Elasticity',
            color_continuous_scale='RdYlGn_r',
            title='Average Price Elasticity by Department',
            labels={'Avg_Elasticity': 'Price Elasticity', 'DEPARTMENT': 'Department'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        segment_counts = filtered_df['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig2 = px.pie(
            segment_counts,
            names='Segment',
            values='Count',
            title='Distribution of Elasticity Segments',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Department Summary Table**")
    st.dataframe(
        filtered_dept[[
            'DEPARTMENT', 'Products_Analyzed',
            'Avg_Elasticity', 'Avg_Revenue_Uplift', 'Strategic_Priority'
        ]].sort_values('Avg_Revenue_Uplift', ascending=False).round(2),
        use_container_width=True
    )

# TAB 2
with tab2:
    st.subheader("Product Explorer & Price Simulator")
    st.markdown("Select a product and move the slider to simulate revenue impact of price changes.")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_dept = st.selectbox(
            "Select Department",
            sorted(filtered_df['DEPARTMENT'].dropna().unique())
        )
        dept_products = filtered_df[filtered_df['DEPARTMENT'] == selected_dept]
        selected_product = st.selectbox(
            "Select Product",
            sorted(dept_products['COMMODITY_DESC'].dropna().unique())
        )
        product_data = dept_products[dept_products['COMMODITY_DESC'] == selected_product].iloc[0]
        st.markdown("**Product Metrics**")
        st.metric("Price Elasticity", f"{product_data['elasticity']:.3f}")
        st.metric("Segment", product_data['Segment'])
        st.metric("Recommendation", product_data['Recommendation'])
        st.metric("Current Avg Price", f"${product_data['avg_price']:.2f}")
        st.metric("Optimal Price", f"${product_data['optimal_price']:.2f}")
        st.metric("Max Revenue Uplift", f"{product_data['revenue_uplift_pct']:.1f}%")
    with col2:
        price_change = st.slider("Simulate Price Change (%)", min_value=-50, max_value=50, value=0, step=1)
        multiplier = 1 + price_change / 100
        new_price = product_data['avg_price'] * multiplier
        new_qty = product_data['avg_quantity'] * (multiplier ** product_data['elasticity'])
        new_revenue = new_price * new_qty
        current_revenue = product_data['avg_price'] * product_data['avg_quantity']
        revenue_delta = new_revenue - current_revenue
        revenue_delta_pct = (revenue_delta / current_revenue) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Price", f"${new_price:.2f}", delta=f"{price_change}%")
        m2.metric("Expected Quantity", f"{new_qty:.0f}", delta=f"{((new_qty / product_data['avg_quantity']) - 1) * 100:.1f}%")
        m3.metric("Revenue Impact", f"${revenue_delta:+,.2f}", delta=f"{revenue_delta_pct:.1f}%")
        price_range_pct = np.linspace(-50, 50, 200)
        revenues = []
        for pct in price_range_pct:
            m = 1 + pct / 100
            p = product_data['avg_price'] * m
            q = product_data['avg_quantity'] * (m ** product_data['elasticity'])
            revenues.append(p * q)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range_pct, y=revenues, mode='lines', name='Revenue', line=dict(color='steelblue', width=2)))
        fig.add_vline(x=0, line_dash='dash', line_color='gray', annotation_text='Current Price')
        fig.add_vline(x=product_data['price_change_pct'], line_dash='dash', line_color='green', annotation_text='Optimal Price')
        fig.add_vline(x=price_change, line_dash='dot', line_color='red', annotation_text='Your Selection')
        fig.update_layout(title='Revenue Curve — How Revenue Changes with Price', xaxis_title='Price Change (%)', yaxis_title='Expected Revenue ($)', height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3
with tab3:
    st.subheader("Top Pricing Opportunities")
    st.markdown("Products with the highest revenue uplift potential at optimal price point.")
    rec_filter = st.multiselect(
        "Filter by Recommendation",
        options=sorted(filtered_df['Recommendation'].dropna().unique()),
        default=sorted(filtered_df['Recommendation'].dropna().unique())
    )
    opp_df = filtered_df[filtered_df['Recommendation'].isin(rec_filter)]
    top_opp = opp_df.nlargest(20, 'revenue_uplift_pct')[[
        'COMMODITY_DESC', 'DEPARTMENT', 'elasticity',
        'Segment', 'Recommendation', 'avg_price',
        'optimal_price', 'revenue_uplift_pct'
    ]].round(3)
    top_opp.columns = [
        'Product', 'Department', 'Elasticity', 'Segment',
        'Recommendation', 'Current Price ($)', 'Optimal Price ($)', 'Revenue Uplift (%)'
    ]
    def color_recommendation(val):
        if 'INCREASE' in str(val):
            return 'background-color: #d4edda'
        elif 'REDUCE' in str(val):
            return 'background-color: #f8d7da'
        elif 'PROMOTIONS' in str(val):
            return 'background-color: #fff3cd'
        return ''
    st.dataframe(
        top_opp.style.map(color_recommendation, subset=['Recommendation']),
        use_container_width=True,
        height=400
    )
    fig = px.bar(
        top_opp.head(10),
        x='Revenue Uplift (%)',
        y='Product',
        color='Recommendation',
        orientation='h',
        title='Top 10 Products by Revenue Uplift Potential',
        color_discrete_map={
            'INCREASE PRICE': '#28a745',
            'HOLD PRICE': '#fd7e14',
            'USE PROMOTIONS': '#ffc107',
            'REDUCE PRICE': '#dc3545'
        }
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Demand Curve Visualizer")
    st.markdown("Select a product to see its empirical demand curve and optimal price point.")
    selected_dc = st.selectbox(
        "Select Product",
        sorted(filtered_df['COMMODITY_DESC'].dropna().unique()),
        key='demand_curve_select'
    )
    product_info = filtered_df[filtered_df['COMMODITY_DESC'] == selected_dc].iloc[0]
    price_range = np.linspace(product_info['avg_price'] * 0.4, product_info['avg_price'] * 2.0, 300)
    qty_range = product_info['avg_quantity'] * ((price_range / product_info['avg_price']) ** product_info['elasticity'])
    product_weekly = weekly[weekly['PRODUCT_ID'] == product_info['PRODUCT_ID']]
    fig = go.Figure()
    if len(product_weekly) > 0:
        fig.add_trace(go.Scatter(
            x=product_weekly['Avg_Price'],
            y=product_weekly['Total_Quantity'],
            mode='markers',
            name='Actual Weekly Data',
            marker=dict(color='lightblue', size=7, opacity=0.7)
        ))
    fig.add_trace(go.Scatter(
        x=price_range, y=qty_range,
        mode='lines', name='Fitted Demand Curve',
        line=dict(color='steelblue', width=2.5)
    ))
    fig.add_vline(x=product_info['avg_price'], line_dash='dash', line_color='gray', annotation_text=f"Current: ${product_info['avg_price']:.2f}")
    fig.add_vline(x=product_info['optimal_price'], line_dash='dash', line_color='green', annotation_text=f"Optimal: ${product_info['optimal_price']:.2f}")
    fig.update_layout(title=f'Demand Curve: {selected_dc}', xaxis_title='Price ($)', yaxis_title='Quantity Sold per Week', height=500, legend=dict(x=0.7, y=0.95))
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price Elasticity", f"{product_info['elasticity']:.3f}")
    col2.metric("Segment", product_info['Segment'])
    col3.metric("Recommendation", product_info['Recommendation'])
    col4.metric("Revenue Uplift Potential", f"{product_info['revenue_uplift_pct']:.1f}%")

# TAB 5
with tab5:
    st.subheader("Strategic Portfolio Assessment")
    st.markdown("*Pricing intelligence reframed as an asset evaluation framework — invest, optimize, or divest.*")

    st.markdown("#### Category Attractiveness Matrix")
    st.markdown("Each bubble is a product. Size = revenue uplift opportunity. Position = strategic value.")

    filtered_df_plot = filtered_df.copy()
    filtered_df_plot['log_revenue'] = np.log1p(filtered_df_plot['avg_revenue'])
    med_log_revenue = filtered_df_plot['log_revenue'].median()

    fig = px.scatter(
        filtered_df_plot,
        x='log_revenue',
        y='elasticity',
        size='revenue_uplift_pct',
        color='DEPARTMENT',
        hover_name='COMMODITY_DESC',
        hover_data={
            'elasticity': ':.3f',
            'avg_revenue': ':,.2f',
            'revenue_uplift_pct': ':.1f',
            'Recommendation': True,
            'Segment': True,
            'log_revenue': False
        },
        title='Category Attractiveness Matrix — Revenue Size vs Pricing Power',
        labels={
            'log_revenue': 'Revenue Size (log scale)',
            'elasticity': 'Pricing Power  ←  (less negative = stronger)  →'
        },
        size_max=20
    )

    fig.add_hline(y=-1.0, line_dash='dash', line_color='gray', line_width=1)
    fig.add_vline(x=med_log_revenue, line_dash='dash', line_color='gray', line_width=1)

    fig.add_annotation(xref='paper', yref='paper', x=0.85, y=0.95, text="⭐ Core Assets — Protect", showarrow=False, font=dict(color='green', size=11), bgcolor='rgba(255,255,255,0.7)')
    fig.add_annotation(xref='paper', yref='paper', x=0.15, y=0.95, text="🚀 Growth — Invest", showarrow=False, font=dict(color='blue', size=11), bgcolor='rgba(255,255,255,0.7)')
    fig.add_annotation(xref='paper', yref='paper', x=0.85, y=0.05, text="⚙️ Mature — Optimize", showarrow=False, font=dict(color='orange', size=11), bgcolor='rgba(255,255,255,0.7)')
    fig.add_annotation(xref='paper', yref='paper', x=0.15, y=0.05, text="⚠️ Weak — Review", showarrow=False, font=dict(color='red', size=11), bgcolor='rgba(255,255,255,0.7)')

    fig.update_layout(height=580)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Value Creation by Department")
    priority_emoji = {'INVEST': '🟢', 'OPTIMIZE': '🟡', 'REVIEW': '🔴'}
    display_summary = filtered_dept.copy()
    display_summary['Priority'] = display_summary['Strategic_Priority'].map(lambda x: f"{priority_emoji.get(x, '')} {x}")
    display_summary = display_summary[['DEPARTMENT', 'Products_Analyzed', 'Total_Revenue', 'Avg_Elasticity', 'Avg_Revenue_Uplift', 'Priority']].sort_values('Avg_Revenue_Uplift', ascending=False)
    display_summary.columns = ['Department', 'Products', 'Total Revenue ($)', 'Avg Elasticity', 'Avg Revenue Uplift (%)', 'Strategic Priority']
    st.dataframe(display_summary.round(2), use_container_width=True, height=380)

    st.markdown("#### Executive Summary")
    top_dept = filtered_dept.nlargest(1, 'Avg_Revenue_Uplift')['DEPARTMENT'].values[0]
    avg_opportunity = filtered_dept['Avg_Revenue_Uplift'].mean()
    core_assets = len(filtered_df[filtered_df['elasticity'] > -0.5])
    invest_depts = filtered_dept[filtered_dept['Strategic_Priority'] == 'INVEST']['DEPARTMENT'].tolist()
    review_depts = filtered_dept[filtered_dept['Strategic_Priority'] == 'REVIEW']['DEPARTMENT'].tolist()
    optimize_depts = filtered_dept[filtered_dept['Strategic_Priority'] == 'OPTIMIZE']['DEPARTMENT'].tolist()

    st.info(f"""
**Strategic Assessment Summary**

Analysis of **{len(filtered_df):,} product categories** across **{filtered_df['DEPARTMENT'].nunique()} departments** identifies an average revenue optimization opportunity of **{avg_opportunity:.1f}%** through systematic price repositioning.

**{top_dept}** represents the highest value creation opportunity across the portfolio.

**{core_assets} categories** demonstrate strong pricing power — inelastic demand signals defensible revenue streams and represents priority candidates for portfolio protection or acquisition targeting.

---

**Recommended Strategic Actions:**

🟢 **Invest & Protect** — {', '.join(invest_depts) if invest_depts else 'None identified'}

🟡 **Optimize** — {', '.join(optimize_depts) if optimize_depts else 'None identified'}

🔴 **Review for Rationalization** — {', '.join(review_depts) if review_depts else 'None identified'}
    """)