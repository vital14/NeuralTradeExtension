import requests
import pandas as pd
import matplotlib.pyplot as plt

class EconomicDataFetcherUsd:
    def __init__(self, alpha_vantage_api_key, fred_api_key):
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.fred_api_key = fred_api_key
        self.data_sources = {
            'interest_rates': {'US': self.get_us_interest_rates},
            'inflation_rate': {'US': self.get_us_inflation_rate},
            'gdp': {'US': self.get_us_gdp},
            'trade_balance': {'US': self.get_us_trade_balance},
            'government_debt': {'US': self.get_us_government_debt}
        }
    
    def get_us_interest_rates(self):
        url = f'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={self.alpha_vantage_api_key}'
        response = requests.get(url)
        return response.json()

    def get_us_inflation_rate(self):
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={self.fred_api_key}&file_type=json'
        response = requests.get(url)
        return response.json()

    def get_us_gdp(self):
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={self.fred_api_key}&file_type=json'
        response = requests.get(url)
        return response.json()

    def get_us_trade_balance(self):
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id=BOPGSTB&api_key={self.fred_api_key}&file_type=json'
        response = requests.get(url)
        return response.json()

    def get_us_government_debt(self):
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id=GFDEBTN&api_key={self.fred_api_key}&file_type=json'
        response = requests.get(url)
        return response.json()

    def fetch_data(self, data_type, country):
        if data_type in self.data_sources and country in self.data_sources[data_type]:
            return self.data_sources[data_type][country]()
        else:
            raise ValueError(f"No data source available for {data_type} in {country}")

    def clean_data(self, data, key='observations', value_key='value'):
        if key in data:
            df = pd.DataFrame(data[key])
        else:
            raise KeyError(f"Expected key '{key}' not found in data: {data}")

        df[value_key] = pd.to_numeric(df[value_key], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        return df

    def analyze_currency(self, country):
        try:
            interest_rates = self.fetch_data('interest_rates', country)
            inflation_rate = self.fetch_data('inflation_rate', country)
            gdp = self.fetch_data('gdp', country)
            trade_balance = self.fetch_data('trade_balance', country)
            government_debt = self.fetch_data('government_debt', country)

            interest_rates_df = self.clean_data(interest_rates, key='data', value_key='value')
            inflation_rate_df = self.clean_data(inflation_rate)
            gdp_df = self.clean_data(gdp)
            trade_balance_df = self.clean_data(trade_balance)
            government_debt_df = self.clean_data(government_debt)

            print("Interest Rates:\n", interest_rates_df.tail())
            print("Inflation Rate:\n", inflation_rate_df.tail())
            print("GDP:\n", gdp_df.tail())
            print("Trade Balance:\n", trade_balance_df.tail())
            print("Government Debt:\n", government_debt_df.tail())

            plt.figure(figsize=(14, 8))

            plt.subplot(3, 2, 1)
            plt.plot(interest_rates_df['date'], interest_rates_df['value'])
            plt.title('Interest Rates')

            plt.subplot(3, 2, 2)
            plt.plot(inflation_rate_df['date'], inflation_rate_df['value'])
            plt.title('Inflation Rate')

            plt.subplot(3, 2, 3)
            plt.plot(gdp_df['date'], gdp_df['value'])
            plt.title('GDP')

            plt.subplot(3, 2, 4)
            plt.plot(trade_balance_df['date'], trade_balance_df['value'])
            plt.title('Trade Balance')

            plt.subplot(3, 2, 5)
            plt.plot(government_debt_df['date'], government_debt_df['value'])
            plt.title('Government Debt')

            plt.tight_layout()
            plt.show()
        except ValueError as e:
            print(f"Data retrieval error: {e}")
        except KeyError as e:
            print(f"Data format error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

alpha_vantage_api_key = 'TWXPZAGYFIH9HMAL'
fred_api_key = '66cf3e68310f57a0bd5f874d7f9a1b1a'

fetcher = EconomicDataFetcherUsd(alpha_vantage_api_key, fred_api_key)
country = 'US'  
fetcher.analyze_currency(country)
