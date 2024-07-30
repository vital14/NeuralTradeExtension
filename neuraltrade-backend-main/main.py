from api.oanda_api import OandaApi
from infrastructure.instrument_collection import instrumentCollection
from simulation.ma_cross import run_ma_sim
from dateutil import parser
from infrastructure.collect_data import run_collection

if __name__ == '__main__':
    api = OandaApi()  
 #   instrumentCollection.CreateFile(api.get_account_instruments(), "./data")
 #   instrumentCollection.LoadInstruments("./data")    
 #   run_collection(instrumentCollection, api)       
    """
      como tem muito pares se foi rodar para criar novo pares e nao perder muito tempo e espaco vc pode diminiur qtd nesse vetor e       
      our_curr = ["AUD", "CAD", "JPY", "USD", "EUR", "GBP", "NZD"]
      e tempos nisso
      granularity in ["M5", "H1", "H4"]:

    """
 #   run_ma_sim()        para criar rodar algorimo de moving avg
    