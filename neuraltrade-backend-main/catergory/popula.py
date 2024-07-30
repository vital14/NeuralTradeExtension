from Category import Category, Factor, DatabaseHandler

economic_indicators = Category('Economic Indicators')
political_events = Category('Political Events')
market_sentiment = Category('Market Sentiment')
natural_events = Category('Natural Events')
other_factors = Category('Other Factors')

# Create factors
gdp = Factor('GDP', economic_indicators, 'Gross Domestic Product')
elections = Factor('Elections', political_events, 'National elections')
investor_confidence = Factor('Investor Confidence', market_sentiment, 'Market confidence of investors')
earthquake = Factor('Earthquake', natural_events, 'Natural disaster affecting a region')
interest_rates = Factor('Interest Rates', other_factors, 'Central bank interest rates')

# Save to database
db_handler = DatabaseHandler()

db_handler.add_factor(gdp)
db_handler.add_factor(elections)
db_handler.add_factor(investor_confidence)
db_handler.add_factor(earthquake)
db_handler.add_factor(interest_rates)

db_handler.close()
