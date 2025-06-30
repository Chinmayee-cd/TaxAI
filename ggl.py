import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from cryptography.fernet import Fernet
import openai

openai.api_key = st.secrets["openai"]["api_key"]

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def chat_with_tax_ai(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful tax assistant. Provide accurate and concise information about tax-related queries."},
                {"role": "user", "content": user_input}])
        return response.choices[0].message['content']
    except Exception as error_message:
        return f"Something went wrong: {str(error_message)}"

def create_encryption_key():
    return Fernet.generate_key()

def lock_data(enc_key, raw_text):
    cipher_tool = Fernet(enc_key)
    return cipher_tool.encrypt(raw_text.encode())

def unlock_data(enc_key, encrypted_text):
    cipher_tool = Fernet(enc_key)
    return cipher_tool.decrypt(encrypted_text).decode()

import requests
from bs4 import BeautifulSoup
import re

def get_tax_rates():
    irs_url = "https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2025"
    response = requests.get(irs_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    irs_text = soup.find('div', class_='text-long').text
    brackets = []
    for line in irs_text.split("\n"):
        match = re.search(r"(\d+)% for incomes over \$(\d+[\d,]*)", line)
        if match:
            tax_percentage = int(match.group(1)) / 100
            income_threshold = int(match.group(2).replace(",", ""))
            brackets.append((income_threshold, tax_percentage))
    brackets.append((0, 0.10))
    brackets = sorted(brackets, key=lambda x: x[0])
    return brackets

try:
    TAX_RATES = get_tax_rates()
except:
    TAX_RATES = [(10000, 0.10), (30000, 0.15), (70000, 0.20), (150000, 0.25), (float('inf'), 0.30)]

def calculate_income_tax(earnings):
    total_tax = 0
    prev_bracket = 0
    for bracket_limit, tax_rate in TAX_RATES:
        if earnings > prev_bracket:
            taxable_amount = min(earnings - prev_bracket, bracket_limit - prev_bracket)
            total_tax += taxable_amount * tax_rate
            prev_bracket = bracket_limit
        else:
            break
    return total_tax

def flag_audit_cases(tax_data):
    columns_needed = ['Income', 'Reported Tax', 'Calculated Tax', 'Discrepancy']
    tax_data["Calculated Tax"] = tax_data["Income"].apply(calculate_income_tax)
    tax_data["Discrepancy"] = abs(tax_data["Calculated Tax"] - tax_data["Reported Tax"])
    data_features = tax_data[columns_needed]
    data_scaler = StandardScaler()
    scaled_data = data_scaler.fit_transform(data_features)
    audit_model = IsolationForest(contamination=0.1, random_state=42)
    tax_data['Flagged'] = audit_model.fit_predict(scaled_data)
    return tax_data[tax_data['Flagged'] == -1]  #-1 indicates suspicious cases

def find_deductions(earnings, expenses):
    deduction_list = []
    if expenses['medical'] > 0.075 * earnings:
        deduction_list.append(('Medical Expenses', expenses['medical'] - 0.075 * earnings))
    if expenses['charitable'] > 0:
        deduction_list.append(('Charitable Contributions', expenses['charitable']))
    if expenses['mortgage_interest'] > 0:
        deduction_list.append(('Mortgage Interest', expenses['mortgage_interest']))
    return deduction_list

def extract_tax_updates(text_update):
    words = word_tokenize(text_update.lower())
    ignored_words = set(stopwords.words('english'))
    processed_words = [word for word in words if word not in ignored_words]
    keywords_to_track = ['deduction', 'credit', 'rate', 'limit', 'increase', 'decrease']
    key_tax_changes = [word for word in processed_words if word in keywords_to_track]
    return key_tax_changes

def check_ssn_format(ssn):
    ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
    return bool(re.match(ssn_pattern, ssn))

def tax_filing_system():
    st.subheader("Tax Filing Portal")
    encryption_key = create_encryption_key()
    user_ssn = st.text_input("Enter SSN (XXX-XX-XXXX):")
    if user_ssn and not check_ssn_format(user_ssn):
        st.error("SSN format invalid! Please enter as XXX-XX-XXXX.")
        return
    if user_ssn:
        encrypted_ssn = lock_data(encryption_key, user_ssn)
    user_name = st.text_input("Enter Full Name:")
    birth_date = st.date_input("Date of Birth:")
    yearly_income = st.number_input("Total Yearly Earnings:", min_value=0.0)
    st.subheader("Deductions")
    medical_expense = st.number_input("Medical Costs:", min_value=0.0)
    charity_donations = st.number_input("Charitable Donations:", min_value=0.0)
    mortgage_cost = st.number_input("Mortgage Interest:", min_value=0.0)
    
    if st.button("Calculate Taxes"):
        raw_tax = calculate_income_tax(yearly_income)
        expense_details = {
            'medical': medical_expense,
            'charitable': charity_donations,
            'mortgage_interest': mortgage_cost
        }
        applied_deductions = find_deductions(yearly_income, expense_details)
        total_deductions = sum(amount for _, amount in applied_deductions)
        net_taxable_income = max(0, yearly_income - total_deductions)
        final_tax = calculate_income_tax(net_taxable_income)
        st.success("Tax Computation Done!")
        st.write(f"Total Earnings: ${yearly_income:.2f}")
        st.write(f"Total Deductions: ${total_deductions:.2f}")
        st.write(f"Taxable Earnings: ${net_taxable_income:.2f}")
        st.write(f"Total Tax Due: ${final_tax:.2f}")
        if applied_deductions:
            st.subheader("Deductions Considered:")
            for deduction, amount in applied_deductions:
                st.write(f"- {deduction}: ${amount:.2f}")
        if st.button("Submit Tax Return"):
            st.success("Tax return filed successfully!")
            st.write("Encrypted SSN:", encrypted_ssn)
            st.write("Decrypted SSN:", unlock_data(encryption_key, encrypted_ssn))
            st.write("Save a copy of this info!")

def validate_csv(df):
    missing_fields = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"The following required fields are missing from the CSV: {', '.join(missing_fields)}")
    if uploaded_data:
        tax_df = pd.read_csv(uploaded_data)
        flagged_cases = flag_audit_cases(tax_df)
        st.write("Flagged Returns for Audit:", len(flagged_cases))
        st.dataframe(flagged_cases)
def audit_selection():
    st.subheader("Audit Selection")
    
    st.write("Required fields in the CSV file:")
    for field in REQUIRED_FIELDS:
        st.write(f"- {field}")
    
    uploaded_file = st.file_uploader("Upload a CSV file of tax returns", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            validate_csv(df)
            
            st.write("Uploaded Data:", df.head())

            # AI-enhanced audit selection
            audit_cases = ai_select_for_audit(df)
            st.write("Returns Selected for Audit (AI-enhanced):", len(audit_cases))
            st.dataframe(audit_cases)

            # Download audited data
            csv = audit_cases.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Audit Report", data=csv, file_name="ai_audit_results.csv", mime="text/csv")
        
        except ValueError as e:
            st.error(f"Error in CSV file: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
# UI
st.title("TaxAI")
selected_feature = st.sidebar.selectbox("Pick a Feature", ["Tax Filing", "Audit Selection", "Tax Law Updates", "Tax Assistant"])

if selected_feature == "Tax Filing":
    tax_filing_system()
elif selected_feature == "Audit Selection":
    REQUIRED_FIELDS = ['SSN', 'Name', 'Income', 'Reported Tax']
    audit_selection()

elif selected_feature == "Tax Law Updates":
    st.subheader("Tax Law Update Scanner")
    tax_text = st.text_area("Paste Tax Law Update:")
    if st.button("Analyze Update"):
        changes_detected = extract_tax_updates(tax_text)
        st.write("Notable Updates:", ", ".join(changes_detected))
elif selected_feature == "Tax Assistant":
    user_query = st.text_input("Ask a tax-related question:")
    if user_query:
        answer = chat_with_tax_ai(user_query)
        st.write("AI Tax Assistant:", answer)
