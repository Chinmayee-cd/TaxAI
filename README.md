# TaxAI

TaxAI is a tax management system for the future. It is an advanced system that applies artificial intelligence and machine learning to automate or simplify all the key processes in tax administration and filing. This system is made up of the following five key features:
User input- allows the users to input their personal information, income and deductions, and other information required for tax filing.
Calculating your tax liability based on the current tax rates from the IRS- this feature applies the current tax rates for the IRS to compute the total tax liability.
Tax deductions- this system automatically applies all tax deductions and gives a detailed account of the final tax liability.
The system can encrypt all sensitive information like your Social Security Number.
This feature uses machine learning, specifically the Isolation Forest algorithm to create cases for audits.

The system employs the use of machine learning to detect any abnormal tax liabilities that do not fall within the expected range. This system also detects any inconsistencies between your tax returns and the actual tax liabilities you should pay. The latter is probably what the IRS used to audit and send you a notice to produce more documents for your tax year.
The system can also upload many files such as CSVs of your tax returns to compute their tax liabilities and the software can also track all the changes in the tax policies by reading all the changes in tax laws and automatically updating the changes that it may deem to be important.
This system uses web scraping to obtain the current tax rates from the IRS website, it may encrypt all sensitive information, use machine learning to select all the audit cases required and use natural language processing to read all the records and interpret them. This software is written in Python and uses the Streamlit web framework, which makes it easy for all the users to access it via their web browsers.
This software is designed to optimize all the tax processes while providing information to the IRS.

## Run the App Locally
1. Install libraries 
```
pip install streamlit pandas numpy scikit-learn nltk cryptography openai requests beautifulsoup4
```
2. Run the app
```
streamlit run ggl.py
```
