import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union


class TradingEnvironment(gym.Env):
    """Custom Environment for financial trading that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000.0, 
                 max_steps: int = None, window_size: int = 10,
                 commission: float = 0.0001, slippage: float = 0.0001,
                 reward_scaling: float = 0.05, position_size: float = 1.0,
                 stop_loss_pct: Optional[float] = 0.02, take_profit_pct: Optional[float] = 0.04,
                 inactivity_penalty: float = -0.00005, bankruptcy_penalty: float = -1.0,
                 drawdown_factor: float = 0.2, win_rate_bonus: float = 0.0005,
                 normalize_rewards: bool = True, capital_efficiency_bonus: float = 0.0,
                 time_decay_factor: float = 0.0):
        """
        Initialize the trading environment
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            initial_balance: Starting account balance
            max_steps: Maximum number of steps per episode. If None, uses dataset length
            window_size: Number of past bars to include in the state
            commission: Commission per trade as percentage
            slippage: Slippage per trade as percentage
            reward_scaling: Scaling factor for rewards to help RL training
            position_size: Base position size as percentage of account balance
            stop_loss_pct: Optional stop loss percentage (None to disable)
            take_profit_pct: Optional take profit percentage (None to disable)
            inactivity_penalty: Small penalty applied each step to discourage excessive waiting
            bankruptcy_penalty: Severe penalty if balance drops to zero
            drawdown_factor: Penalty factor based on current drawdown
            win_rate_bonus: Bonus reward for maintaining high win rate
            normalize_rewards: Whether to normalize rewards relative to initial balance
            capital_efficiency_bonus: Small bonus for keeping capital employed in the market
            time_decay_factor: Discount factor for rewards based on trade duration
        """
        super(TradingEnvironment, self).__init__()
        
        # Validate and process the dataframe
        required_columns = ['open', 'high', 'low', 'close', 'ema_short', 'ema_long', 'atr', 'adx']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataframe must contain column: {col}")
        
        # Store the data and parameters
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Store new reward parameters
        self.inactivity_penalty = inactivity_penalty
        self.bankruptcy_penalty = bankruptcy_penalty
        self.drawdown_factor = drawdown_factor
        self.win_rate_bonus = win_rate_bonus
        self.normalize_rewards = normalize_rewards
        self.capital_efficiency_bonus = capital_efficiency_bonus
        self.time_decay_factor = time_decay_factor
        
        # Set the maximum number of steps
        self.max_steps = len(df) - window_size if max_steps is None else min(max_steps, len(df) - window_size)
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized OHLC, technical indicators, and account state
        self.feature_dim = 8  # Price data plus technical indicators
        self.account_dim = 3  # Position (0=none, 1=long, -1=short), entry price, unrealized pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.feature_dim + self.account_dim), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        super().reset(seed=seed)
        
        # Reset internal pointers
        self.current_step = 0
        self.total_steps = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.entry_idx = 0
        self.profit_loss = 0.0
        self.total_profit_loss = 0.0
        self.trade_count = 0
        self.successful_trades = 0
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.stop_activated = False
        self.take_profit_activated = False
        
        # Trading stats for evaluation
        self.trade_history = []
        self.balance_history = [self.initial_balance]
        
        # Get initial state
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
        
        Returns:
            observation, reward, done, truncated, info
        """
        # Map the action: 0 = Hold, 1 = Buy, 2 = Sell
        trade_action = 0  # Default: Hold
        if action == 1:
            trade_action = 1  # Buy
        elif action == 2:
            trade_action = -1  # Sell
        
        # Get current market data
        prev_price = self.df.iloc[self.current_step + self.window_size - 1]['close']
        current_price = self.df.iloc[self.current_step + self.window_size]['close']
        
        # Initialize reward components dictionary
        reward_components = {
            'inactivity': self.inactivity_penalty,  # Base penalty for each step
            'pnl': 0.0,                             # P&L component (updated during trading)
            'drawdown': 0.0,                        # Drawdown penalty
            'win_rate': 0.0,                        # Win rate bonus
            'capital_efficiency': 0.0,              # Capital efficiency bonus
            'time_decay': 0.0                       # Time decay factor
        }
        
        # Trade tracking
        trade_closed = False
        position_held_steps = 0
        trade_pnl = 0.0
        
        # Initialize flags for stop loss and take profit
        sl_tp_triggered = False
        sl_price = 0.0
        tp_price = 0.0
        
        # Calculate stop loss and take profit levels if we have a position
        if self.position != 0:
            # Track how long the position has been held
            position_held_steps = self.current_step - self.entry_idx
            
            # Apply capital efficiency bonus if enabled
            if self.capital_efficiency_bonus > 0:
                reward_components['capital_efficiency'] = self.capital_efficiency_bonus
            
            if self.stop_loss_pct is not None:
                if self.position == 1:  # Long position
                    sl_price = self.entry_price * (1 - self.stop_loss_pct)
                    if current_price <= sl_price:
                        sl_tp_triggered = True
                        self.stop_activated = True
                else:  # Short position
                    sl_price = self.entry_price * (1 + self.stop_loss_pct)
                    if current_price >= sl_price:
                        sl_tp_triggered = True
                        self.stop_activated = True
            
            if self.take_profit_pct is not None:
                if self.position == 1:  # Long position
                    tp_price = self.entry_price * (1 + self.take_profit_pct)
                    if current_price >= tp_price:
                        sl_tp_triggered = True
                        self.take_profit_activated = True
                else:  # Short position
                    tp_price = self.entry_price * (1 - self.take_profit_pct)
                    if current_price <= tp_price:
                        sl_tp_triggered = True
                        self.take_profit_activated = True
        
        # Handle the stop loss or take profit first if triggered
        if sl_tp_triggered:
            exit_price = current_price
            if self.stop_activated:
                exit_price = sl_price  # Use the stop price if stop was triggered
            elif self.take_profit_activated:
                exit_price = tp_price  # Use the take profit price if TP was triggered
            
            # Calculate P&L for this trade
            trade_pnl = 0
            if self.position == 1:  # Closing a long position
                trade_pnl = (exit_price - self.entry_price) / self.entry_price - self.commission - self.slippage
            else:  # Closing a short position
                trade_pnl = (self.entry_price - exit_price) / self.entry_price - self.commission - self.slippage
            
            # Scale the P&L to account balance
            position_value = self.balance * self.position_size
            pnl_amount = position_value * trade_pnl * self.position
            
            # Update account
            self.profit_loss = pnl_amount
            self.balance += pnl_amount
            self.total_profit_loss += pnl_amount
            
            # Record trade
            self.trade_count += 1
            if pnl_amount > 0:
                self.successful_trades += 1
            
            # Log the trade
            trade_info = {
                'entry_time': self.df.index[self.entry_idx],
                'exit_time': self.df.index[self.current_step + self.window_size],
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'position': self.position,
                'pnl': pnl_amount,
                'balance': self.balance,
                'exit_reason': 'stop_loss' if self.stop_activated else 'take_profit',
                'trade_duration': position_held_steps
            }
            self.trade_history.append(trade_info)
            
            # Reset position
            self.position = 0
            self.entry_price = 0
            self.profit_loss = 0
            self.stop_activated = False
            self.take_profit_activated = False
            
            # Calculate normalized P&L if enabled
            if self.normalize_rewards:
                normalized_pnl = pnl_amount / self.initial_balance
            else:
                normalized_pnl = pnl_amount
            
            # Apply time decay if enabled
            time_decay = 0
            if self.time_decay_factor > 0:
                time_decay = -self.time_decay_factor * position_held_steps
            
            # Set PnL component
            reward_components['pnl'] = normalized_pnl * self.reward_scaling
            reward_components['time_decay'] = time_decay
            
            trade_closed = True
        
        # Process new trade action if we don't have a position or are changing position
        if not sl_tp_triggered and ((self.position == 0 and trade_action != 0) or 
                                    (self.position != 0 and trade_action != 0 and self.position != trade_action)):
            # Close existing position if any
            if self.position != 0:
                # Calculate P&L for this trade
                trade_pnl = 0
                if self.position == 1:  # Closing a long position
                    trade_pnl = (current_price - self.entry_price) / self.entry_price - self.commission - self.slippage
                else:  # Closing a short position
                    trade_pnl = (self.entry_price - current_price) / self.entry_price - self.commission - self.slippage
                
                # Scale the P&L to account balance
                position_value = self.balance * self.position_size
                pnl_amount = position_value * trade_pnl * self.position
                
                # Update account
                self.profit_loss = pnl_amount
                self.balance += pnl_amount
                self.total_profit_loss += pnl_amount
                
                # Record trade
                self.trade_count += 1
                if pnl_amount > 0:
                    self.successful_trades += 1
                
                # Log the trade
                trade_info = {
                    'entry_time': self.df.index[self.entry_idx],
                    'exit_time': self.df.index[self.current_step + self.window_size],
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl_amount,
                    'balance': self.balance,
                    'exit_reason': 'signal',
                    'trade_duration': position_held_steps
                }
                self.trade_history.append(trade_info)
                
                # Calculate normalized P&L if enabled
                if self.normalize_rewards:
                    normalized_pnl = pnl_amount / self.initial_balance
                else:
                    normalized_pnl = pnl_amount
                
                # Apply time decay if enabled
                time_decay = 0
                if self.time_decay_factor > 0:
                    time_decay = -self.time_decay_factor * position_held_steps
                
                # Set PnL component
                reward_components['pnl'] = normalized_pnl * self.reward_scaling
                reward_components['time_decay'] = time_decay
                
                # Reset position before new entry
                self.position = 0
                self.entry_price = 0
                self.profit_loss = 0
                
                trade_closed = True
            
            # Enter new position if requested
            if trade_action != 0:
                # Set entry price with slippage
                if trade_action == 1:  # Long position
                    self.entry_price = current_price * (1 + self.slippage)
                else:  # Short position
                    self.entry_price = current_price * (1 - self.slippage)
                
                self.position = trade_action
                self.entry_idx = self.current_step + self.window_size
                
                # Apply commission
                commission_cost = self.balance * self.position_size * self.commission
                self.balance -= commission_cost
                
                # Log the trade entry
                trade_info = {
                    'entry_time': self.df.index[self.entry_idx],
                    'entry_price': self.entry_price,
                    'position': self.position,
                    'balance': self.balance,
                    'entry_reason': 'signal'
                }
                self.trade_history.append(trade_info)
        
        # Calculate unrealized P&L if we have a position
        unrealized_pnl = 0
        if self.position != 0:
            if self.position == 1:  # Long position
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
            # Scale to account value
            unrealized_pnl = unrealized_pnl * self.balance * self.position_size
        
        # Track max balance and drawdown
        current_portfolio_value = self.balance + unrealized_pnl
        self.balance_history.append(current_portfolio_value)
        
        if current_portfolio_value > self.max_balance:
            self.max_balance = current_portfolio_value
        
        # Calculate current drawdown and add to reward components
        current_drawdown = (self.max_balance - current_portfolio_value) / self.max_balance if self.max_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Add drawdown penalty to reward components
        reward_components['drawdown'] = -current_drawdown * self.drawdown_factor
        
        # Add win rate bonus if applicable
        if self.win_rate_bonus > 0 and self.trade_count > 0:
            win_rate = self.successful_trades / self.trade_count
            if win_rate > 0.5:  # Only give bonus for win rates > 50%
                reward_components['win_rate'] = self.win_rate_bonus * (win_rate - 0.5) * 100
        
        # Move to the next step
        self.current_step += 1
        self.total_steps += 1
        
        # Check if the episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Bankrupt check
        if self.balance <= 0:
            done = True
            reward_components['pnl'] = self.bankruptcy_penalty  # Severe penalty for bankruptcy
        
        # Calculate total reward
        reward = sum(reward_components.values())
        
        # Get the new observation
        observation = self._get_observation()
        
        # Create info dict for monitoring
        win_rate = self.successful_trades / max(1, self.trade_count)
        
        info = {
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_profit_loss,
            'trade_count': self.trade_count,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'current_position': self.position,
            'step': self.current_step,
            'total_steps': self.total_steps,
            'reward_components': reward_components,
            'current_drawdown': current_drawdown
        }
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        """
        Construct the observation from the current market state
        
        Returns:
            numpy array with the observation
        """
        # Calculate the starting point for the window
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size
        
        # Get the window of price data
        window_data = self.df.iloc[start_idx:end_idx]
        
        # Extract features
        features = []
        for idx, row in window_data.iterrows():
            # Calculate normalization factor
            current_price = row['close']
            price_norm = current_price * 0.01  # Scale factor for normalization
            
            # Create feature vector with normalized values
            bar_features = np.array([
                (row['ema_short'] - current_price) / price_norm,  # Distance from short EMA
                (row['ema_long'] - current_price) / price_norm,   # Distance from long EMA
                (row['ema_short'] - row['ema_long']) / price_norm,# EMA crossover
                row['atr'] / price_norm,                          # Normalized ATR
                row['adx'] / 100.0,                               # Normalized ADX
                (row['close'] - row['open']) / price_norm,        # Candle body
                (row['high'] - row['close']) / price_norm,        # Upper wick
                (row['close'] - row['low']) / price_norm          # Lower wick
            ])
            
            # Account state
            if self.position == 0:
                account_state = np.array([0, 0, 0])  # No position
            elif self.position == 1:  # Long
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
                account_state = np.array([1, self.entry_price / current_price, unrealized_pnl])
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price if self.entry_price > 0 else 0
                account_state = np.array([-1, self.entry_price / current_price, unrealized_pnl])
            
            # Combine features with account state
            combined_features = np.concatenate([bar_features, account_state])
            features.append(combined_features)
        
        return np.array(features, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render a visualization of the environment"""
        if mode != 'human':
            return
        
        # Create a simple trading view
        plt.figure(figsize=(15, 10))
        
        # Price chart with trades
        plt.subplot(3, 1, 1)
        plt.plot(self.df['close'], label='Close Price')
        plt.plot(self.df['ema_short'], label=f'EMA Short')
        plt.plot(self.df['ema_long'], label=f'EMA Long')
        
        # Plot trades
        for trade in self.trade_history:
            if 'exit_time' in trade:  # Completed trade
                if trade['position'] == 1:  # Long trade
                    plt.scatter(trade['entry_time'], trade['entry_price'], color='green', marker='^', s=100)
                    plt.scatter(trade['exit_time'], trade['exit_price'], color='red', marker='v', s=100)
                    plt.plot([trade['entry_time'], trade['exit_time']], 
                             [trade['entry_price'], trade['exit_price']], color='blue', linestyle='--')
                else:  # Short trade
                    plt.scatter(trade['entry_time'], trade['entry_price'], color='red', marker='v', s=100)
                    plt.scatter(trade['exit_time'], trade['exit_price'], color='green', marker='^', s=100)
                    plt.plot([trade['entry_time'], trade['exit_time']], 
                             [trade['entry_price'], trade['exit_price']], color='orange', linestyle='--')
        
        plt.title('Price Chart with Trades')
        plt.legend()
        
        # Equity curve
        plt.subplot(3, 1, 2)
        plt.plot(self.balance_history, label='Account Balance')
        plt.title('Equity Curve')
        plt.legend()
        
        # Trade statistics
        plt.subplot(3, 1, 3)
        plt.axis('off')
        stats_text = f"""
        Initial Balance: ${self.initial_balance:.2f}
        Final Balance: ${self.balance:.2f}
        Total P&L: ${self.total_profit_loss:.2f}
        Return: {(self.balance / self.initial_balance - 1) * 100:.2f}%
        Trade Count: {self.trade_count}
        Win Rate: {self.successful_trades / max(1, self.trade_count) * 100:.2f}%
        Max Drawdown: {self.max_drawdown * 100:.2f}%
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12)
        
        plt.tight_layout()
        plt.show()