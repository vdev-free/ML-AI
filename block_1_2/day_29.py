import pandas as pd

df = pd.DataFrame({
    'product': ['Apple', 'Banana', 'apple juice', 'BANANA SHAKE', 'Avocado'],
    'category': ['fruit', 'fruit', 'drink', 'drink', 'fruit']
})

# df['product'] =df['product'].str.lower()
# df['category'] = df['category'].str.upper()



# print(df['product'].str[-4:])

df['product_lower'] = df['product'].str.lower()


print(df['product'].str.contains('banana', case=False))

print(df['product'].str.replace('shake', 'SMOOTHIE', case=False))

print(df['product'].str[:3])