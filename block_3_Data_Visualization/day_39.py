import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
    'shop_a_sales': [100, 110, 120, 130, 140, 150, 160, 170, 160, 150, 140, 130, 120, 110, 100],
    'shop_b_sales': [90, 100, 95, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
})

df['shop_a_smooth'] = df['shop_a_sales'].rolling(window=3).mean()
df['shop_b_smooth'] = df['shop_b_sales'].rolling(window=3).mean()

df_melted = df.melt(id_vars='date',
                    value_vars=['shop_a_sales', 'shop_b_sales', 'shop_a_smooth', 'shop_b_smooth'],
                    var_name='shop',
                    value_name='sales')

fig = px.line(
   df_melted,
   x='date',
   y='sales',
   color='shop',
   markers=True,
   title='Інтерактивний графік продажів магазинів'
)

fig.show()