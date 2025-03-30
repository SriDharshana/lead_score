import pandas as pd

num_customers = 100 
df = pd.DataFrame({
    'Customer_ID': range(501, 501 + num_customers),
    'Age': [25, 30, 35, 40] * (num_customers // 4), 
    'Monthly_Income': [40000, 50000, 60000, 70000] * (num_customers // 4),
    'Past_Purchases': [2, 5, 10, 15] * (num_customers // 4),
    'Last_Interaction_Days': [1, 3, 7, 14] * (num_customers // 4),
    'Lead_Source': ['Website', 'Email', 'Referral', 'Social Media'] * (num_customers // 4),
})

df.to_csv("customer_data_2.csv", index=False)

print(df.head()) 
