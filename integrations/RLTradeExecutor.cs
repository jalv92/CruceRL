using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RLTradeExecutor : Strategy
    {
        #region Variables
        private RLCommunicationIndicator communicationIndicator;
        private bool isIndicatorAttached = false;
        
        // Trading variables
        private MarketPosition currentPosition = MarketPosition.Flat;
        private int currentSignal = 0; // -1 = short, 0 = flat, 1 = long
        private int currentEMA = 0; // 0 = none, 1 = short, 2 = long
        private double currentPositionSize = 1.0;
        private double currentStopLoss = 0;
        private double currentTakeProfit = 0;
        private double basePositionSize = 1; // Default contract size
        private int orderRetryCount = 3;
        private int orderRetryDelayMs = 1000;
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Base Position Size", Description = "Base number of contracts/units for trades", Order = 4, GroupName = "Trading Settings")]
        public double BasePositionSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Retry Count", Description = "Number of times to retry failed orders", Order = 5, GroupName = "Trading Settings")]
        public int OrderRetryCount { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Use Managed Orders", Description = "Use managed or unmanaged order handling", Order = 6, GroupName = "Trading Settings")]
        public bool UseManagedOrders { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Trading strategy that uses the RLCommunicationIndicator to execute trades based on Python RL model signals";
                Name = "RLTradeExecutor";
                
                // Default settings
                BasePositionSize = 1;
                OrderRetryCount = 3;
                UseManagedOrders = true; // Default to managed orders
                
                // Set up calculation mode
                Calculate = Calculate.OnBarClose;
                BarsRequiredToTrade = 20;
                IsExitOnSessionCloseStrategy = true;
                
                // Set this according to UseManagedOrders
                IsUnmanaged = !UseManagedOrders;
            }
            else if (State == State.Configure)
            {
                // Set local variables from properties
                basePositionSize = BasePositionSize;
                orderRetryCount = OrderRetryCount;
                
                // Update IsUnmanaged based on UseManagedOrders
                IsUnmanaged = !UseManagedOrders;
            }
            else if (State == State.DataLoaded)
            {
                // Add our communication indicator
                communicationIndicator = new RLCommunicationIndicator();
                AddChartIndicator(communicationIndicator);
                isIndicatorAttached = true;
                
                // Register for trade signals
                communicationIndicator.OnTradeSignalReceived += HandleTradeSignal;
            }
            else if (State == State.Terminated)
            {
                // Cleanup
                if (isIndicatorAttached && communicationIndicator != null)
                {
                    communicationIndicator.OnTradeSignalReceived -= HandleTradeSignal;
                }
            }
        }

        protected override void OnBarUpdate()
        {
            // No need to implement any trading logic here
            // All trading is triggered via the indicator callback
        }

        private void HandleTradeSignal(int signal, int emaChoice, double positionSize, double stopLoss, double takeProfit)
        {
            try
            {
                // Store current trading parameters
                currentSignal = signal;
                currentEMA = emaChoice;
                currentPositionSize = positionSize;
                currentStopLoss = stopLoss;
                currentTakeProfit = takeProfit;
                
                Print($"Received trade signal: Signal={signal}, EMA={emaChoice}, Size={positionSize}, " +
                      $"SL={stopLoss.ToString("F2")}, TP={takeProfit.ToString("F2")}");
                
                // Execute the trading decision
                ExecuteTradingDecision();
            }
            catch (Exception ex)
            {
                Print($"Error handling trade signal: {ex.Message}");
            }
        }

        private void ExecuteTradingDecision()
        {
            try
            {
                // Get current position
                MarketPosition currentPosition = Position.MarketPosition;
                
                // Calculate actual position size
                int quantity = (int)Math.Ceiling(basePositionSize * currentPositionSize);
                quantity = Math.Max(1, quantity); // Ensure at least 1 contract
                
                // Calculate stop loss and take profit prices
                double currentPrice = Close[0];
                double stopLossPrice = 0;
                double takeProfitPrice = 0;
                
                if (currentSignal == 1) // Long
                {
                    stopLossPrice = currentPrice - (currentStopLoss * TickSize);
                    takeProfitPrice = currentPrice + (currentTakeProfit * TickSize);
                }
                else if (currentSignal == -1) // Short
                {
                    stopLossPrice = currentPrice + (currentStopLoss * TickSize);
                    takeProfitPrice = currentPrice - (currentTakeProfit * TickSize);
                }
                
                // Log trading decision
                Print($"Executing trade: Signal={currentSignal}, EMA={currentEMA}, Size={quantity}, " +
                      $"SL={stopLossPrice.ToString("F2")}, TP={takeProfitPrice.ToString("F2")}");
                
                // Choose between managed and unmanaged order execution based on settings
                if (UseManagedOrders)
                {
                    ExecuteManagedOrders(currentPosition, currentSignal, quantity, stopLossPrice, takeProfitPrice);
                }
                else
                {
                    ExecuteUnmanagedOrders(currentPosition, currentSignal, quantity, stopLossPrice, takeProfitPrice);
                }
            }
            catch (Exception ex)
            {
                Print("Error executing trading decision: " + ex.Message);
            }
        }
        
        private void ExecuteManagedOrders(MarketPosition currentPosition, int signalDirection, int quantity, 
                                         double stopLossPrice, double takeProfitPrice)
        {
            try
            {
                // Execute order based on current position and signal
                if (currentPosition == MarketPosition.Flat)
                {
                    // No position, potentially enter new position
                    if (signalDirection == 1) // Long signal
                    {
                        EnterLong(quantity, "RL_Long");
                        
                        // Set stop loss and take profit
                        if (currentStopLoss > 0)
                            ExitLongStopMarket(quantity, stopLossPrice, "RL_Long_SL", "RL_Long");
                        
                        if (currentTakeProfit > 0)
                            ExitLongLimit(quantity, takeProfitPrice, "RL_Long_TP", "RL_Long");
                    }
                    else if (signalDirection == -1) // Short signal
                    {
                        EnterShort(quantity, "RL_Short");
                        
                        // Set stop loss and take profit
                        if (currentStopLoss > 0)
                            ExitShortStopMarket(quantity, stopLossPrice, "RL_Short_SL", "RL_Short");
                        
                        if (currentTakeProfit > 0)
                            ExitShortLimit(quantity, takeProfitPrice, "RL_Short_TP", "RL_Short");
                    }
                    // If signal is 0, stay flat
                }
                else if (currentPosition == MarketPosition.Long)
                {
                    // Currently long
                    if (signalDirection <= 0) // Exit signal (flat or short)
                    {
                        // Close all long positions
                        ExitLong("RL_Long_Exit");
                        
                        // If short signal, also enter short
                        if (signalDirection == -1)
                        {
                            EnterShort(quantity, "RL_Short");
                            
                            // Set stop loss and take profit
                            if (currentStopLoss > 0)
                                ExitShortStopMarket(quantity, stopLossPrice, "RL_Short_SL", "RL_Short");
                            
                            if (currentTakeProfit > 0)
                                ExitShortLimit(quantity, takeProfitPrice, "RL_Short_TP", "RL_Short");
                        }
                    }
                    else
                    {
                        // Maintain long position but update SL/TP
                        // First cancel existing orders
                        ExitLong("RL_Long_SL", "RL_Long");
                        ExitLong("RL_Long_TP", "RL_Long");
                        
                        // Set new orders
                        if (currentStopLoss > 0)
                            ExitLongStopMarket(Position.Quantity, stopLossPrice, "RL_Long_SL", "RL_Long");
                        
                        if (currentTakeProfit > 0)
                            ExitLongLimit(Position.Quantity, takeProfitPrice, "RL_Long_TP", "RL_Long");
                    }
                }
                else if (currentPosition == MarketPosition.Short)
                {
                    // Currently short
                    if (signalDirection >= 0) // Exit signal (flat or long)
                    {
                        // Close all short positions
                        ExitShort("RL_Short_Exit");
                        
                        // If long signal, also enter long
                        if (signalDirection == 1)
                        {
                            EnterLong(quantity, "RL_Long");
                            
                            // Set stop loss and take profit
                            if (currentStopLoss > 0)
                                ExitLongStopMarket(quantity, stopLossPrice, "RL_Long_SL", "RL_Long");
                            
                            if (currentTakeProfit > 0)
                                ExitLongLimit(quantity, takeProfitPrice, "RL_Long_TP", "RL_Long");
                        }
                    }
                    else
                    {
                        // Maintain short position but update SL/TP
                        // First cancel existing orders
                        ExitShort("RL_Short_SL", "RL_Short");
                        ExitShort("RL_Short_TP", "RL_Short");
                        
                        // Set new orders
                        if (currentStopLoss > 0)
                            ExitShortStopMarket(Position.Quantity, stopLossPrice, "RL_Short_SL", "RL_Short");
                        
                        if (currentTakeProfit > 0)
                            ExitShortLimit(Position.Quantity, takeProfitPrice, "RL_Short_TP", "RL_Short");
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Error executing managed orders: {ex.Message}");
            }
        }

        private void ExecuteUnmanagedOrders(MarketPosition currentPosition, int signalDirection, int quantity, 
                                           double stopLossPrice, double takeProfitPrice)
        {
            try
            {
                // Execute order based on current position and signal
                if (currentPosition == MarketPosition.Flat)
                {
                    // No position, potentially enter new position
                    if (signalDirection == 1) // Long signal
                    {
                        // Submit entry order
                        SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market, quantity, 0, 0, string.Empty, "RL_Long");
                        
                        // Set stop loss and take profit
                        if (currentStopLoss > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.StopMarket, quantity, 0, stopLossPrice, "RL_Long", "RL_Long_SL");
                        
                        if (currentTakeProfit > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Limit, quantity, takeProfitPrice, 0, "RL_Long", "RL_Long_TP");
                    }
                    else if (signalDirection == -1) // Short signal
                    {
                        // Submit entry order
                        SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, quantity, 0, 0, string.Empty, "RL_Short");
                        
                        // Set stop loss and take profit
                        if (currentStopLoss > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.StopMarket, quantity, 0, stopLossPrice, "RL_Short", "RL_Short_SL");
                        
                        if (currentTakeProfit > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Limit, quantity, takeProfitPrice, 0, "RL_Short", "RL_Short_TP");
                    }
                    // If signal is 0, stay flat
                }
                else if (currentPosition == MarketPosition.Long)
                {
                    // Currently long
                    if (signalDirection <= 0) // Exit signal (flat or short)
                    {
                        // Close all long positions
                        SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, string.Empty, "RL_Long_Exit");
                        
                        // Cancel any pending SL/TP orders
                        CancelOrder("RL_Long_SL");
                        CancelOrder("RL_Long_TP");
                        
                        // If short signal, also enter short
                        if (signalDirection == -1)
                        {
                            // Submit entry order
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, quantity, 0, 0, string.Empty, "RL_Short");
                            
                            // Set stop loss and take profit
                            if (currentStopLoss > 0)
                                SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.StopMarket, quantity, 0, stopLossPrice, "RL_Short", "RL_Short_SL");
                            
                            if (currentTakeProfit > 0)
                                SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Limit, quantity, takeProfitPrice, 0, "RL_Short", "RL_Short_TP");
                        }
                    }
                    else
                    {
                        // Maintain long position but update SL/TP
                        // First cancel existing orders
                        CancelOrder("RL_Long_SL");
                        CancelOrder("RL_Long_TP");
                        
                        // Set new orders
                        if (currentStopLoss > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.StopMarket, Position.Quantity, 0, stopLossPrice, "RL_Long", "RL_Long_SL");
                        
                        if (currentTakeProfit > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Limit, Position.Quantity, takeProfitPrice, 0, "RL_Long", "RL_Long_TP");
                    }
                }
                else if (currentPosition == MarketPosition.Short)
                {
                    // Currently short
                    if (signalDirection >= 0) // Exit signal (flat or long)
                    {
                        // Close all short positions
                        SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market, Position.Quantity, 0, 0, string.Empty, "RL_Short_Exit");
                        
                        // Cancel any pending SL/TP orders
                        CancelOrder("RL_Short_SL");
                        CancelOrder("RL_Short_TP");
                        
                        // If long signal, also enter long
                        if (signalDirection == 1)
                        {
                            // Submit entry order
                            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market, quantity, 0, 0, string.Empty, "RL_Long");
                            
                            // Set stop loss and take profit
                            if (currentStopLoss > 0)
                                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.StopMarket, quantity, 0, stopLossPrice, "RL_Long", "RL_Long_SL");
                            
                            if (currentTakeProfit > 0)
                                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Limit, quantity, takeProfitPrice, 0, "RL_Long", "RL_Long_TP");
                        }
                    }
                    else
                    {
                        // Maintain short position but update SL/TP
                        // First cancel existing orders
                        CancelOrder("RL_Short_SL");
                        CancelOrder("RL_Short_TP");
                        
                        // Set new orders
                        if (currentStopLoss > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.StopMarket, Position.Quantity, 0, stopLossPrice, "RL_Short", "RL_Short_SL");
                        
                        if (currentTakeProfit > 0)
                            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Limit, Position.Quantity, takeProfitPrice, 0, "RL_Short", "RL_Short_TP");
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Error executing unmanaged orders: {ex.Message}");
            }
        }

        // Helper method to cancel orders by name
        private void CancelOrder(string orderName)
        {
            foreach (Order order in Orders)
            {
                if (order.Name == orderName && order.OrderState != OrderState.Filled && order.OrderState != OrderState.Cancelled)
                {
                    try
                    {
                        CancelOrder(order);
                    }
                    catch (Exception ex)
                    {
                        Print($"Error cancelling order {orderName}: {ex.Message}");
                    }
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            try
            {
                // Log execution details
                Print($"Order executed: {marketPosition} {quantity} @ {price:F2}, OrderId: {orderId}");
                
                // Send trade notification back to Python server
                if (communicationIndicator != null)
                {
                    // Determine action and P&L
                    string action = "";
                    double entryPrice = 0;
                    double exitPrice = 0;
                    double pnl = 0;
                    
                    // Check the order name to determine the action
                    if (orderId.Contains("RL_Long") && !orderId.Contains("Exit") && !orderId.Contains("SL") && !orderId.Contains("TP"))
                    {
                        action = "Enter Long";
                        entryPrice = price;
                    }
                    else if (orderId.Contains("RL_Short") && !orderId.Contains("Exit") && !orderId.Contains("SL") && !orderId.Contains("TP"))
                    {
                        action = "Enter Short";
                        entryPrice = price;
                    }
                    else if (orderId.Contains("Exit") || orderId.Contains("SL") || orderId.Contains("TP"))
                    {
                        // This is an exit order
                        if (orderId.Contains("Long"))
                        {
                            action = "Exit Long";
                            exitPrice = price;
                            
                            // Calculate P&L for long position
                            double positionEntryPrice = Position.AveragePrice;
                            pnl = (exitPrice - positionEntryPrice) * quantity * Instrument.MasterInstrument.PointValue;
                        }
                        else if (orderId.Contains("Short"))
                        {
                            action = "Exit Short";
                            exitPrice = price;
                            
                            // Calculate P&L for short position
                            double positionEntryPrice = Position.AveragePrice;
                            pnl = (positionEntryPrice - exitPrice) * quantity * Instrument.MasterInstrument.PointValue;
                        }
                    }
                    
                    // Send execution data to Python
                    if (!string.IsNullOrEmpty(action))
                    {
                        communicationIndicator.SendTradeExecutionUpdate(action, entryPrice, exitPrice, pnl, quantity);
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Error in OnExecutionUpdate: {ex.Message}");
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            try
            {
                if (error != ErrorCode.NoError)
                {
                    Print($"Order error: {error}, OrderId: {order.Id}, Comment: {comment}");
                }
            }
            catch (Exception ex)
            {
                Print($"Error in OnOrderUpdate: {ex.Message}");
            }
        }
    }
}