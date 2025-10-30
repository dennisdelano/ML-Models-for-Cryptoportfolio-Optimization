# main.py
import numpy as np
import pandas as pd
from AlgorithmImports import *

class Xgboost4(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2025, 5, 17)
        self.SetCash(100000)
        
        # Add Bitcoin with Trade data (not Quote data) to get Volume
        crypto = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance)
        self.symbol = crypto.Symbol
        # Request Trade bars instead of Quote bars for volume data
        crypto.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add VIX data
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        
        # Rolling windows for feature calculation
        self.price_window = RollingWindow[float](200)
        self.volume_window = RollingWindow[float](200)
        self.vix_window = RollingWindow[float](200)
        
        # Store historical data for training
        self.training_data = []
        self.lookback_hours = 180 * 24  # 180 days
        
        # Model and retraining schedule
        self.model = None
        self.last_train_date = None
        self.retrain_days = 30
        self.warmup_complete = False
        
        # Custom loss hyperparameters
        self.sharpe_weight = 1.0  # Sharpe ratio weight (α)
        self.mse_weight = 0.5     # MSE weight (β)
        self.l2_reg = 0.01        # L2 regularization weight (λ)
        
        # Feature names for consistency
        self.feature_columns = [
            'close', 'high', 'low', 'open', 'volume', 'vix_close',
            'return_pct1', 'return_pct2', 'return_pct5',
            'ema12', 'ema26', 'macd', 'macd_signal', 'macd_hist',
            'vix_ema10', 'vix_ema50',
            'sma20', 'std20', 'bb_upper', 'bb_lower',
            'vol_mean20', 'vol_std20', 'vol_zscore20', 'rsi14'
        ]
        
        # Warm up with 200 days
        self.SetWarmUp(timedelta(days=200))
        
        # Schedule training
        self.Schedule.On(
            self.DateRules.MonthStart(self.symbol),
            self.TimeRules.At(0, 0),
            self.TrainModel
        )
        
    def OnData(self, data):
        if self.IsWarmingUp:
            self._UpdateWindows(data)
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.TrainModel()
            
        # Update rolling windows
        self._UpdateWindows(data)
        
        # Need enough data for features
        if not self.price_window.IsReady or not self.volume_window.IsReady:
            return
        
        # Only trade on Bitcoin data
        if not data.ContainsKey(self.symbol):
            return
            
        # Calculate features for current bar
        features = self._CalculateFeatures()
        if features is None:
            return
            
        # Make prediction
        if self.model is not None:
            try:
                import xgboost as xgb
                
                X = np.array([features])
                # Convert to DMatrix for xgb.Booster prediction
                dtest = xgb.DMatrix(X)
                prediction = self.model.predict(dtest)[0]
                
                # Trading logic with threshold
                threshold = 0.0001  # 0.01% minimum predicted return
                
                if prediction > threshold and not self.Portfolio[self.symbol].Invested:
                    self.SetHoldings(self.symbol, 0.95)
                    self.Debug(f"{self.Time} BUY - Predicted: {prediction:.6f}")
                elif prediction < -threshold and self.Portfolio[self.symbol].Invested:
                    self.Liquidate(self.symbol)
                    self.Debug(f"{self.Time} SELL - Predicted: {prediction:.6f}")
                    
            except Exception as e:
                self.Debug(f"Prediction error: {str(e)}")
                
        # Store data point for future training
        if features is not None and data.ContainsKey(self.symbol):
            bar = data[self.symbol]
            self.training_data.append({
                'time': self.Time,
                'features': features,
                'close': bar.Close
            })
            
            # Keep only lookback_hours of data
            if len(self.training_data) > self.lookback_hours + 50:
                self.training_data = self.training_data[-self.lookback_hours:]
    
    def _UpdateWindows(self, data):
        """Update rolling windows with new data"""
        if data.ContainsKey(self.symbol):
            bar = data[self.symbol]
            self.price_window.Add(bar.Close)
            # Use Volume from TradeBar, or default to 1 if not available
            if hasattr(bar, 'Volume') and bar.Volume > 0:
                self.volume_window.Add(bar.Volume)
            else:
                # Fallback: use a default volume if not available
                self.volume_window.Add(1.0)
            
        if data.ContainsKey(self.vix):
            vix_bar = data[self.vix]
            self.vix_window.Add(vix_bar.Close)
    
    def _CalculateFeatures(self):
        """Calculate all features for the current bar"""
        try:
            # Convert windows to lists
            prices = [self.price_window[i] for i in range(min(200, self.price_window.Count))]
            volumes = [self.volume_window[i] for i in range(min(200, self.volume_window.Count))]
            
            if len(prices) < 50 or len(volumes) < 50:
                return None
                
            # Get VIX value (forward-filled)
            vix_close = self.vix_window[0] if self.vix_window.Count > 0 else 15.0
            
            # Current values
            close = prices[0]
            high = max(prices[:24]) if len(prices) >= 24 else close
            low = min(prices[:24]) if len(prices) >= 24 else close
            open_price = prices[23] if len(prices) >= 24 else close
            volume = volumes[0]
            
            # Returns
            return_pct1 = (prices[0] / prices[1] - 1) if len(prices) > 1 else 0
            return_pct2 = (prices[0] / prices[2] - 1) if len(prices) > 2 else 0
            return_pct5 = (prices[0] / prices[5] - 1) if len(prices) > 5 else 0
            
            # MACD
            ema12 = self._ema(prices[:50], 12)
            ema26 = self._ema(prices[:50], 26)
            macd = ema12 - ema26
            
            # For MACD signal, we need historical MACD values
            macd_values = []
            for i in range(min(9, len(prices) - 26)):
                e12 = self._ema(prices[i:min(i+50, len(prices))], 12)
                e26 = self._ema(prices[i:min(i+50, len(prices))], 26)
                macd_values.append(e12 - e26)
            macd_signal = self._ema(macd_values, 9) if len(macd_values) >= 9 else macd
            macd_hist = macd - macd_signal
            
            # VIX EMAs
            vix_values = [self.vix_window[i] for i in range(min(50, self.vix_window.Count))]
            vix_ema10 = self._ema(vix_values, 10) if len(vix_values) >= 10 else vix_close
            vix_ema50 = self._ema(vix_values, 50) if len(vix_values) >= 50 else vix_close
            
            # Bollinger Bands
            sma20 = np.mean(prices[:20]) if len(prices) >= 20 else close
            std20 = np.std(prices[:20]) if len(prices) >= 20 else 0
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            
            # Volume z-score
            vol_mean20 = np.mean(volumes[:20]) if len(volumes) >= 20 else volume
            vol_std20 = np.std(volumes[:20]) if len(volumes) >= 20 else 1
            vol_zscore20 = (volume - vol_mean20) / (vol_std20 + 1e-9)
            
            # RSI
            rsi14 = self._calculate_rsi(prices[:50], 14) if len(prices) >= 50 else 50
            
            features = [
                close, high, low, open_price, volume, vix_close,
                return_pct1, return_pct2, return_pct5,
                ema12, ema26, macd, macd_signal, macd_hist,
                vix_ema10, vix_ema50,
                sma20, std20, bb_upper, bb_lower,
                vol_mean20, vol_std20, vol_zscore20, rsi14
            ]
            
            return features
            
        except Exception as e:
            self.Debug(f"Error calculating features: {str(e)}")
            return None
    
    def _ema(self, values, period):
        """Calculate exponential moving average"""
        if len(values) < period:
            return values[0] if len(values) > 0 else 0
        alpha = 2 / (period + 1)
        ema = values[-1]
        for i in range(len(values) - 2, max(len(values) - period - 1, -1), -1):
            ema = alpha * values[i] + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _custom_loss_gradient(self, y_pred, dtrain):
        """
        Custom loss gradient: L = -α·Sharpe + β·MSE(y,ŷ) + λ·||w||²
        
        Gradient computation:
        ∂L/∂ŷ = -α·∂Sharpe/∂ŷ + β·∂MSE/∂ŷ
        
        where:
        - Sharpe = mean(returns) / std(returns)
        - ∂Sharpe/∂ŷ_i ≈ (1/σ - μ·(y_i - μ)/σ³) / n
        - ∂MSE/∂ŷ_i = 2(ŷ_i - y_i) / n
        """
        y_true = dtrain.get_label()
        n = len(y_true)
        
        # MSE gradient component
        mse_grad = 2 * (y_pred - y_true) / n
        
        # Sharpe gradient component
        # Treat predictions as returns for Sharpe calculation
        mean_return = np.mean(y_pred)
        std_return = np.std(y_pred) + 1e-8  # Add small epsilon to avoid division by zero
        
        # ∂Sharpe/∂ŷ_i = (1/σ - μ·(ŷ_i - μ)/σ³) / n
        sharpe_grad = ((1 / std_return) - (mean_return * (y_pred - mean_return) / (std_return ** 3))) / n
        
        # Combined gradient: -α·∂Sharpe/∂ŷ + β·∂MSE/∂ŷ
        grad = -self.sharpe_weight * sharpe_grad + self.mse_weight * mse_grad
        
        # Hessian (second derivative) - approximate with constant for simplicity
        hess = np.ones(n) * (self.mse_weight * 2 / n + self.sharpe_weight / (n * std_return))
        
        return grad, hess
    
    def _custom_loss_metric(self, y_pred, dtrain):
        """
        Custom evaluation metric for monitoring
        Returns the actual loss value (not used for optimization, just monitoring)
        """
        y_true = dtrain.get_label()
        
        # MSE component
        mse = np.mean((y_pred - y_true) ** 2)
        
        # Sharpe component (negative because we want to maximize Sharpe)
        mean_return = np.mean(y_pred)
        std_return = np.std(y_pred) + 1e-8
        sharpe = mean_return / std_return
        
        # Combined loss
        loss = -self.sharpe_weight * sharpe + self.mse_weight * mse
        
        return 'custom_loss', loss
    
    def TrainModel(self):
        """Train XGBoost model with custom loss function"""
        if len(self.training_data) < 1000:
            self.Debug(f"Not enough data to train: {len(self.training_data)} points")
            return
            
        try:
            import xgboost as xgb
            
            self.Debug(f"Training model with {len(self.training_data)} data points")
            
            # Prepare training data
            X_list = []
            y_list = []
            
            for i in range(len(self.training_data) - 1):
                features = self.training_data[i]['features']
                # Target is next hour's return
                current_close = self.training_data[i]['close']
                next_close = self.training_data[i + 1]['close']
                target = (next_close / current_close) - 1
                
                X_list.append(features)
                y_list.append(target)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X, label=y)
            
            # XGBoost parameters with L2 regularization (lambda)
            params = {
                'max_depth': 5,
                'eta': 0.05,  # learning_rate
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'lambda': self.l2_reg,  # L2 regularization
                'objective': 'reg:squarederror',  # Will be overridden by custom objective
                'seed': 42
            }
            
            # Train with custom objective function
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=150,
                obj=self._custom_loss_gradient,
                custom_metric=self._custom_loss_metric,
                verbose_eval=False
            )
            
            self.last_train_date = self.Time
            self.Debug(f"Model trained successfully at {self.Time} with custom Sharpe-MSE loss")
            
            # Log some metrics
            y_pred = self.model.predict(dtrain)
            mse = np.mean((y_pred - y) ** 2)
            sharpe = np.mean(y_pred) / (np.std(y_pred) + 1e-8)
            self.Debug(f"Training MSE: {mse:.6f}, Pred Sharpe: {sharpe:.4f}")
            
        except Exception as e:
            self.Debug(f"Error training model: {str(e)}")
            import traceback
            self.Debug(f"Traceback: {traceback.format_exc()}")
    
    def OnEndOfAlgorithm(self):
        """Summary at end of backtest"""
        self.Debug(f"=== Backtest Summary ===")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        # Calculate basic stats
        if self.Portfolio.TotalPortfolioValue > 0:
            total_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000 * 100
            self.Debug(f"Total Return: {total_return:.2f}%")
        
        self.Debug(f"Training data points collected: {len(self.training_data)}")