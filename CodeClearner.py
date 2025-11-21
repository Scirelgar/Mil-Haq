import re

# 1. Convert the 'Date' column in TrainData to datetime objects
TrainData['Date'] = pd.to_datetime(TrainData['Date'], format='%d/%m/%Y')

# 2. Melt the TrainData DataFrame from its wide format to a long format
# Identify columns that are not 'Date' for melting
id_vars = ['Date']
value_vars = [col for col in TrainData.columns if col not in id_vars]

restructured_df = pd.melt(TrainData, id_vars=id_vars, value_vars=value_vars,
                          var_name='Tenor_Maturity_Str', value_name='Price (Y)')

# 3. Extract the 'Tenor (T)' and 'Maturity (τ)' values from the 'Tenor_Maturity_Str' column
def extract_tenor_maturity(s):
    tenor_match = re.search(r'Tenor : (\d+)', s)
    maturity_match = re.search(r'Maturity : ([\d.]+)', s)
    tenor = int(tenor_match.group(1)) if tenor_match else None
    maturity = float(maturity_match.group(1)) if maturity_match else None
    return tenor, maturity

restructured_df[['Tenor (T)', 'Maturity (τ)']] = restructured_df['Tenor_Maturity_Str'].apply(lambda x: pd.Series(extract_tenor_maturity(x)))

# 4. Convert the extracted 'Tenor (T)' column to an integer data type and the 'Maturity (τ)' column to a float data type.
# 'Tenor (T)' is already int due to extraction logic, but ensure it
restructured_df['Tenor (T)'] = restructured_df['Tenor (T)'].astype(int)
# 'Maturity (τ)' is already float due to extraction logic, but ensure it
restructured_df['Maturity (τ)'] = restructured_df['Maturity (τ)'].astype(float)

# 5. Ensure the 'Price (Y)' column is of float data type.
restructured_df['Price (Y)'] = restructured_df['Price (Y)'].astype(float)

# 6. Select and reorder the columns to match the target_df's column order
final_restructured_df = restructured_df[['Date', 'Tenor (T)', 'Maturity (τ)', 'Price (Y)']]
final_restructured_df.info()