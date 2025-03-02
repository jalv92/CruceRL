using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.Net.Sockets;
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
    public class RLExecutor : Strategy
    {
        #region TCP Server Manager
        /// <summary>
        /// Helper class to manage TCP server functionality
        /// </summary>
        private class TcpServerManager
        {
            // Properties
            public string Name { get; private set; }
            public string ServerIP { get; private set; }
            public int Port { get; private set; }
            public bool IsRunning { get; private set; }
            public DateTime LastHeartbeatTime { get; set; }
            public bool IsConnected => Client != null && Client.Connected;
            
            // Connection objects
            public TcpListener Server { get; private set; }
            public TcpClient Client { get; private set; }
            public NetworkStream ClientStream { get; private set; }
            
            // Thread management
            private Thread serverThread;
            private Queue<string> messageBuffer;
            private int bufferSize;
            
            // Delegates and events
            public delegate void MessageReceivedHandler(string message);
            public event MessageReceivedHandler OnMessageReceived;
            
            // Connection timeout
            private int connectionTimeout;
            
            // Logger function (using Print from NinjaTrader)
            private Action<string> logFunc;
            
            /// <summary>
            /// Constructor for TcpServerManager
            /// </summary>
            public TcpServerManager(string name, string serverIP, int port, int bufferSize = 60, int connectionTimeout = 5, Action<string> logger = null)
            {
                Name = name;
                ServerIP = serverIP;
                Port = port;
                this.bufferSize = bufferSize;
                this.connectionTimeout = connectionTimeout;
                messageBuffer = new Queue<string>();
                LastHeartbeatTime = DateTime.MinValue;
                logFunc = logger ?? (message => { /* Default empty logger */ });
            }
            
            /// <summary>
            /// Start the TCP server
            /// </summary>
            public bool Start()
            {
                if (IsRunning)
                    return true;
                    
                try
                {
                    // Clean up existing resources
                    Stop();
                    
                    // Log server start attempt
                    logFunc($"Starting {Name} server on {ServerIP}:{Port}...");
                    
                    // Create and start TCP listener
                    IPAddress ipAddress = IPAddress.Parse(ServerIP);
                    Server = new TcpListener(ipAddress, Port);
                    Server.Start();
                    
                    // Log successful start
                    logFunc($"{Name} server started on {ServerIP}:{Port}");
                    
                    // Start server thread
                    IsRunning = true;
                    serverThread = new Thread(ServerLoop);
                    serverThread.IsBackground = true;
                    serverThread.Start();
                    
                    return true;
                }
                catch (Exception ex)
                {
                    // Log error and clean up
                    logFunc($"Error starting {Name} server: {ex.Message}");
                    
                    if (Server != null)
                    {
                        try { Server.Stop(); } catch { }
                        Server = null;
                    }
                    
                    IsRunning = false;
                    return false;
                }
            }
            
            /// <summary>
            /// Stop the TCP server and clean up resources
            /// </summary>
            public void Stop()
            {
                try
                {
                    // Mark server as stopping
                    IsRunning = false;
                    
                    // Close client stream
                    if (ClientStream != null)
                    {
                        try
                        {
                            ClientStream.Flush();
                            ClientStream.Close();
                        }
                        catch (Exception ex)
                        {
                            logFunc($"Error closing {Name} client stream: {ex.Message}");
                        }
                        finally
                        {
                            ClientStream = null;
                        }
                    }
                    
                    // Close client connection
                    if (Client != null)
                    {
                        try
                        {
                            Client.Close();
                        }
                        catch (Exception ex)
                        {
                            logFunc($"Error closing {Name} client: {ex.Message}");
                        }
                        finally
                        {
                            Client = null;
                        }
                    }
                    
                    // Stop the server
                    if (Server != null)
                    {
                        try
                        {
                            Server.Stop();
                        }
                        catch (Exception ex)
                        {
                            logFunc($"Error stopping {Name} server: {ex.Message}");
                        }
                        finally
                        {
                            Server = null;
                        }
                    }
                    
                    // Wait for thread to terminate
                    if (serverThread != null && serverThread.IsAlive)
                    {
                        try
                        {
                            serverThread.Join(2000);
                            if (serverThread.IsAlive)
                            {
                                // Last resort
                                serverThread.Abort();
                            }
                        }
                        catch { }
                        serverThread = null;
                    }
                    
                    logFunc($"{Name} server stopped successfully");
                }
                catch (Exception ex)
                {
                    logFunc($"Error stopping {Name} server: {ex.Message}");
                }
                finally
                {
                    // Force resource cleanup
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }
            
            /// <summary>
            /// Send a message to the connected client or buffer it
            /// </summary>
            public bool SendMessage(string message)
            {
                try
                {
                    // Check if connected to a client
                    if (Client != null && Client.Connected && ClientStream != null)
                    {
                        // Send data directly
                        byte[] data = Encoding.ASCII.GetBytes(message);
                        ClientStream.Write(data, 0, data.Length);
                        
                        // Log non-heartbeat messages for debugging
                        if (!message.StartsWith("PING") && !message.StartsWith("PONG") && !message.StartsWith("HEARTBEAT"))
                        {
                            logFunc($"[{Name} Server] Sent: {message.Trim()}");
                        }
                        
                        return true;
                    }
                    else
                    {
                        // Buffer the message if it's not a heartbeat
                        if (!message.StartsWith("PING") && !message.StartsWith("PONG") && !message.StartsWith("HEARTBEAT"))
                        {
                            messageBuffer.Enqueue(message);
                            
                            // Limit buffer size
                            while (messageBuffer.Count > bufferSize)
                                messageBuffer.Dequeue();
                                
                            logFunc($"[{Name} Server] Buffered message (no connection): {message.Trim()}");
                        }
                        
                        return false;
                    }
                }
                catch (Exception ex)
                {
                    logFunc($"Error sending {Name} message: {ex.Message}");
                    
                    // Buffer non-heartbeat messages for retry
                    if (!message.StartsWith("PING") && !message.StartsWith("PONG") && !message.StartsWith("HEARTBEAT"))
                    {
                        messageBuffer.Enqueue(message);
                        
                        // Limit buffer size
                        while (messageBuffer.Count > bufferSize)
                            messageBuffer.Dequeue();
                    }
                    
                    // Client connection might be broken, clean up
                    if (ClientStream != null)
                    {
                        try { ClientStream.Close(); } catch { }
                        ClientStream = null;
                    }
                    
                    if (Client != null)
                    {
                        try { Client.Close(); } catch { }
                        Client = null;
                    }
                    
                    return false;
                }
            }
            
            /// <summary>
            /// Main server loop to accept and handle client connections
            /// </summary>
            private void ServerLoop()
            {
                logFunc($"{Name} server thread started. Waiting for Python client connection...");
                
                while (IsRunning)
                {
                    try
                    {
                        // Check for pending connections
                        if (Server.Pending())
                        {
                            // Accept the client connection
                            Client = Server.AcceptTcpClient();
                            Client.NoDelay = true;
                            Client.SendBufferSize = 65536;
                            Client.ReceiveBufferSize = 65536;
                            
                            // Get the client stream
                            ClientStream = Client.GetStream();
                            
                            logFunc($"Python client connected to {Name} server from {Client.Client.RemoteEndPoint}");
                            
                            // Send welcome message
                            string welcomeMessage = Name == "Data" ? "SERVER_READY\n" : "ORDER_SERVER_READY\n";
                            SendMessage(welcomeMessage);
                            
                            // Update heartbeat time
                            LastHeartbeatTime = DateTime.Now;
                            
                            // Send accumulated data from buffer
                            while (messageBuffer.Count > 0)
                            {
                                try
                                {
                                    string data = messageBuffer.Dequeue();
                                    SendMessage(data);
                                }
                                catch (Exception ex)
                                {
                                    logFunc($"Error sending buffered data: {ex.Message}");
                                }
                            }
                            
                            // Process data from the client
                            byte[] buffer = new byte[4096];
                            StringBuilder messageBuilder = new StringBuilder();
                            
                            // Main client data reception loop
                            while (IsRunning && Client != null && Client.Connected)
                            {
                                // Check if there's data to read
                                if (Client.Available > 0)
                                {
                                    int bytesRead = ClientStream.Read(buffer, 0, buffer.Length);
                                    
                                    if (bytesRead == 0)
                                    {
                                        // Connection closed by client
                                        break;
                                    }
                                    
                                    string data = Encoding.ASCII.GetString(buffer, 0, bytesRead);
                                    messageBuilder.Append(data);
                                    
                                    // Process complete messages
                                    string message = messageBuilder.ToString();
                                    int newlineIndex;
                                    
                                    while ((newlineIndex = message.IndexOf('\n')) != -1)
                                    {
                                        // Extract complete message
                                        string completeMessage = message.Substring(0, newlineIndex).Trim();
                                        
                                        // Remove the processed message from buffer
                                        message = message.Substring(newlineIndex + 1);
                                        
                                        // Process the message
                                        if (!string.IsNullOrEmpty(completeMessage))
                                        {
                                            // Handle PING directly here for faster response
                                            if (completeMessage.Equals("PING", StringComparison.OrdinalIgnoreCase))
                                            {
                                                SendMessage("PONG\n");
                                                LastHeartbeatTime = DateTime.Now;
                                            }
                                            else
                                            {
                                                // Notify subscribers about received message
                                                OnMessageReceived?.Invoke(completeMessage);
                                            }
                                        }
                                    }
                                    
                                    // Update buffer with remaining incomplete message
                                    messageBuilder.Clear();
                                    messageBuilder.Append(message);
                                }
                                
                                // Short sleep to prevent high CPU usage
                                Thread.Sleep(10);
                                
                                // Send heartbeat periodically
                                if (DateTime.Now - LastHeartbeatTime > TimeSpan.FromSeconds(5))
                                {
                                    try
                                    {
                                        SendMessage("PING\n");
                                        LastHeartbeatTime = DateTime.Now;
                                    }
                                    catch
                                    {
                                        // Connection might be broken
                                        break;
                                    }
                                }
                            }
                            
                            // Client disconnected or error occurred
                            logFunc($"Python client disconnected from {Name} server");
                            
                            // Clean up client resources
                            if (ClientStream != null)
                            {
                                try { ClientStream.Close(); } catch { }
                                ClientStream = null;
                            }
                            
                            if (Client != null)
                            {
                                try { Client.Close(); } catch { }
                                Client = null;
                            }
                        }
                        
                        // Short sleep before checking for new connections
                        Thread.Sleep(100);
                    }
                    catch (Exception ex)
                    {
                        logFunc($"Error in {Name} server loop: {ex.Message}");
                        
                        // Clean up client resources
                        if (ClientStream != null)
                        {
                            try { ClientStream.Close(); } catch { }
                            ClientStream = null;
                        }
                        
                        if (Client != null)
                        {
                            try { Client.Close(); } catch { }
                            Client = null;
                        }
                        
                        // Short delay before retrying
                        Thread.Sleep(1000);
                    }
                }
                
                logFunc($"{Name} server thread stopped");
            }
        }
        #endregion
        
        #region Variables
        // TCP communication 
        private TcpServerManager dataServer;
        private TcpServerManager orderServer;
        private string listenIP = "127.0.0.1";
        private int dataPort = 5000;
        private int orderPort = 5001;
        private int connectionTimeout = 5;
        private int dataBufferSize = 60;
        private bool connectionErrorLogged = false;
        
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
        [Display(Name = "Server IP", Description = "IP address to listen on", Order = 1, GroupName = "Socket Settings")]
        public string ServerIP { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Data Port", Description = "Port to send market data to Python", Order = 2, GroupName = "Socket Settings")]
        public int DataPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Port", Description = "Port to receive trading signals from Python", Order = 3, GroupName = "Socket Settings")]
        public int OrderPort { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Connection Timeout", Description = "Timeout in seconds for socket connections", Order = 4, GroupName = "Socket Settings")]
        public int ConnectionTimeout { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Base Position Size", Description = "Base number of contracts/units for trades", Order = 5, GroupName = "Trading Settings")]
        public double BasePositionSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Retry Count", Description = "Number of times to retry failed orders", Order = 6, GroupName = "Trading Settings")]
        public int OrderRetryCount { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Use Managed Orders", Description = "Use managed or unmanaged order handling", Order = 7, GroupName = "Trading Settings")]
        public bool UseManagedOrders { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Trading strategy that communicates with a Python RL model for signal generation and execution";
                Name = "RLExecutor";
                
                // Default settings
                ServerIP = "127.0.0.1";
                DataPort = 5000;
                OrderPort = 5001;
                ConnectionTimeout = 5;
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
                listenIP = ServerIP;
                dataPort = DataPort;
                orderPort = OrderPort;
                connectionTimeout = ConnectionTimeout;
                basePositionSize = BasePositionSize;
                orderRetryCount = OrderRetryCount;
                
                // Update IsUnmanaged based on UseManagedOrders
                IsUnmanaged = !UseManagedOrders;
                
                // Initialize TCP server managers
                dataServer = new TcpServerManager("Data", listenIP, dataPort, dataBufferSize, connectionTimeout, Print);
                orderServer = new TcpServerManager("Order", listenIP, orderPort, dataBufferSize, connectionTimeout, Print);
                
                // Register message handlers
                dataServer.OnMessageReceived += ProcessDataMessage;
                orderServer.OnMessageReceived += ProcessOrderMessage;
            }
            else if (State == State.Realtime)
            {
                // Reset connection error flag
                connectionErrorLogged = false;
                
                // Initialize server asynchronously when switching to real-time
                // This prevents freezing the UI while waiting for initialization
                this.Dispatcher.InvokeAsync(() =>
                {
                    // Start the servers
                    Print("Starting data and order servers...");
                    
                    bool dataServerStarted = dataServer.Start();
                    Thread.Sleep(500);
                    bool orderServerStarted = orderServer.Start();
                    
                    if (dataServerStarted && orderServerStarted) {
                        Print("Servers started successfully. Waiting for Python client to connect...");
                    } else {
                        Print("ERROR: Failed to start one or more servers. Python client will not be able to connect.");
                    }
                });
                
                Print("Starting servers asynchronously...");
            }
            else if (State == State.Terminated)
            {
                // Stop servers and close client connections when terminated
                dataServer.Stop();
                orderServer.Stop();
            }
        }

        protected override void OnBarUpdate()
        {
            // No need to implement any bar update logic
            // All trading is triggered via socket communication
        }
        
        #region Message Processing Methods
        private void ProcessDataMessage(string message)
        {
            try
            {
                // Handle incoming messages from Python client
                if (message.Equals("PONG", StringComparison.OrdinalIgnoreCase))
                {
                    // Heartbeat response - handled in server manager
                    return;
                }
                
                if (message.StartsWith("CONNECT:", StringComparison.OrdinalIgnoreCase))
                {
                    // Client sending connection info
                    string[] parts = message.Split(':');
                    if (parts.Length > 1)
                    {
                        string clientInfo = parts[1];
                        Print($"Python client info: {clientInfo}");
                    }
                    
                    // Respond with server info
                    dataServer.SendMessage($"SERVER_INFO:NinjaTrader:{Instrument.MasterInstrument.Name}\n");
                    return;
                }
                
                // Process any data-specific messages here
                Print($"Received from Python data client: {message}");
            }
            catch (Exception ex)
            {
                Print($"Error processing data message: {ex.Message}");
            }
        }
        
        private void ProcessOrderMessage(string message)
        {
            try
            {
                if (string.IsNullOrEmpty(message))
                    return;
                
                // PING/HEARTBEAT handling already done in server manager
                if (message.Equals("PING", StringComparison.OrdinalIgnoreCase) ||
                    message.Equals("PONG", StringComparison.OrdinalIgnoreCase) ||
                    message.Equals("HEARTBEAT", StringComparison.OrdinalIgnoreCase))
                {
                    return;
                }
                
                // Process trading signals from Python (expected format: signal,emaChoice,positionSize,stopLoss,takeProfit)
                ProcessTradingSignal(message);
            }
            catch (Exception ex)
            {
                Print($"Error processing order message: {ex.Message}");
                orderServer.SendMessage("ERROR:Processing_Error\n");
            }
        }
        
        private void ProcessTradingSignal(string message)
        {
            string[] commandParts = message.Split(',');
            
            if (commandParts.Length < 5)
            {
                Print("Invalid order message format. Expected at least 5 parameters.");
                // Send error response to client
                orderServer.SendMessage("ERROR:Invalid_Format\n");
                return;
            }
            
            // Parse parameters
            int tradeSignal;
            int emaChoice;
            double positionSize;
            double stopLoss;
            double takeProfit;
            
            if (!int.TryParse(commandParts[0], out tradeSignal) ||
                !int.TryParse(commandParts[1], out emaChoice) ||
                !double.TryParse(commandParts[2], out positionSize) ||
                !double.TryParse(commandParts[3], out stopLoss) ||
                !double.TryParse(commandParts[4], out takeProfit))
            {
                Print("Invalid order message data. Could not parse values.");
                orderServer.SendMessage("ERROR:Invalid_Values\n");
                return;
            }
            
            // Validate parameters
            if (tradeSignal < -1 || tradeSignal > 1)
            {
                Print("Invalid trade signal: " + tradeSignal);
                orderServer.SendMessage("ERROR:Invalid_Signal\n");
                return;
            }
            
            if (emaChoice < 0 || emaChoice > 2)
            {
                Print("Invalid EMA choice: " + emaChoice);
                orderServer.SendMessage("ERROR:Invalid_EMA\n");
                return;
            }
            
            if (positionSize <= 0 || positionSize > 10)
            {
                Print("Invalid position size: " + positionSize);
                orderServer.SendMessage("ERROR:Invalid_Size\n");
                return;
            }
            
            // Store current trading parameters
            currentSignal = tradeSignal;
            currentEMA = emaChoice;
            currentPositionSize = positionSize;
            currentStopLoss = stopLoss;
            currentTakeProfit = takeProfit;
            
            Print($"Received trade signal: Signal={tradeSignal}, EMA={emaChoice}, Size={positionSize}, " +
                  $"SL={stopLoss.ToString("F2")}, TP={takeProfit.ToString("F2")}");
            
            // Execute the trading decision
            ExecuteTradingDecision();
            
            // Confirm the message was processed correctly
            orderServer.SendMessage($"ORDER_CONFIRMED:{tradeSignal},{emaChoice},{positionSize}\n");
        }
        #endregion
        
        #region Trading Execution Methods
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
                if (!IsUnmanaged)
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
        #endregion
        
        #region Order Event Handlers
        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            try
            {
                // Log execution details
                Print($"Order executed: {marketPosition} {quantity} @ {price:F2}, OrderId: {orderId}");
                
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
                    string executionMessage = $"TRADE_EXECUTED:{action},{entryPrice},{exitPrice},{pnl},{quantity}\n";
                    orderServer.SendMessage(executionMessage);
                    Print($"Sent trade execution data: {action}, P&L: {pnl}");
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
        #endregion
    }
}