# region imports
from AlgorithmImports import *
# endregion


class NeuralNetwork1(QCAlgorithm):
    model = None  # Model object
    model_is_training = False  # Model state flag

    def initialize(self):
        # Locally Lean installs free sample data, to download more data please visit https://www.quantconnect.com/docs/v2/lean-cli/datasets/downloading-data
        self.set_start_date(2024, 1, 1)  # Set Start Date
        self.set_end_date(2025, 9, 12)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        self.btc = self.add_crypto("BTCUSD", Resolution.Daily)

        self.train(self.neural_network_training)

    def neural_network_training(self):
        self.Debug("Training started...")
        self.model_is_training = True

        history = self.history(self.btc.symbol, 365 * 5, self.btc.resolution)

        if history.empty:
            self.Debug("No historical data available for training.")
            self.model_is_training = False
            return

        self.model_is_training = False
        self.Debug("Training finished.")

    def on_data(self, data: Slice):
        """on_data event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        """
        if self.model_is_training:
            return  # Skip if model is training
