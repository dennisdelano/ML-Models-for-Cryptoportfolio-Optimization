# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import xgboost as xgb
import joblib

class CasualFluorescentYellowRabbit(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_cash(100000)
        self.btc = self.add_crypto("BTCUSD", Resolution.DAILY)
        

        model_key = "model"
        self.object_store.contains_key(model_key)
        file_name = self.object_store.get_file_path(model_key)
        self.model = joblib.load(file_name)

    def on_data(self, data: Slice):
        history = self.History(self.btc.Symbol, 6, Resolution.DAILY)
        close = history.loc[:,"close"].copy()
        history["return_pct"] = close.sub(close.shift(periods=1)).div(close.shift(periods=1))
        history.dropna(inplace=True)
        
        features = np.array(history["return_pct"].values)
        #self.Debug(f"{X}")
        
        features_mean = 0.0023283005951960775
        features_std = 0.036133938445466036

        X = (features - features_mean) / features_std

        dmat = xgb.DMatrix(X.reshape(1, -1))

        y_pred = self.model.predict(dmat)

        if not self.portfolio.invested:
            if y_pred > 0.05:
                self.market_order(self.btc.symbol, 1)
                self.Debug(f"Bitcoin wird gekauft da Prediction = {y_pred}")

        if self.portfolio.invested:
            if y_pred < -0.02:
                self.Liquidate()
                self.Debug(f"Bitcoin wird verkauft da Prediction = {y_pred}")
