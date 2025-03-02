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
using System.Xml.Serialization;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;

namespace NinjaTrader.NinjaScript.Indicators
{
    /// <summary>
    /// This indicator handles bidirectional communication with a Python-based RL trading system.
    /// It separates the communication logic from the trading strategy.
    /// </summary>
    public class RLCommunicationIndicator : Indicator
    {
        #region TCP Server Manager
        /// <summary>
        /// Helper class to manage TCP server functionality and eliminate code duplication
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
                            
                            // Send accumulated data from buffer for Data server
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
        // Server components using optimized manager
        private TcpServerManager dataServer;
        private TcpServerManager orderServer;
        private string listenIP = "127.0.0.1";
        private int dataPort = 5000;
        private int orderPort = 5001;
        private int dataBufferSize = 60; // Number of bars to buffer
        private Queue<string> dataBuffer = new Queue<string>();
        private bool isInitialDataSent = false;
        private bool connectionErrorLogged = false; // Prevent log spam
        private int lastMinute = -1;
        private bool waitForFullBar = true;
        
        // Data extraction variables
        private bool isExtractingData = false;
        private bool isExtractionComplete = false;
        private int totalBarsToExtract = 0;
        private int extractedBarsCount = 0;
        private string extractionCommand = "EXTRACT_HISTORICAL_DATA";
        
        // Trading callback delegates
        public delegate void TradeSignalDelegate(int signal, int emaChoice, double positionSize, double stopLoss, double takeProfit);
        public TradeSignalDelegate OnTradeSignalReceived;
        
        // Technical indicators
        private EMA emaShort;
        private EMA emaLong;
        private ATR atr;
        private ADX adx;
        
        // Variable to store the date from which to filter data
        private DateTime? startFromDate = null;
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Server IP", Description = "IP address of the Python server", Order = 1, GroupName = "Socket Settings")]
        public string ServerIP { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Data Port", Description = "Port to send market data to Python", Order = 2, GroupName = "Socket Settings")]
        public int DataPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Port", Description = "Port to receive trading signals from Python", Order = 3, GroupName = "Socket Settings")]
        public int OrderPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Wait For Full Bar", Description = "Wait for full bar to form before sending", Order = 4, GroupName = "Data Settings")]
        public bool WaitForFullBar { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Short Period", Description = "Period for short EMA calculation", Order = 5, GroupName = "Indicators")]
        public int EMAShortPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Long Period", Description = "Period for long EMA calculation", Order = 6, GroupName = "Indicators")]
        public int EMALongPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ATR Period", Description = "Period for ATR calculation", Order = 7, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ADX Period", Description = "Period for ADX calculation", Order = 8, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Historical Bars", Description = "Maximum number of historical bars to extract", Order = 9, GroupName = "Data Settings")]
        public int MaxHistoricalBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Connection Timeout", Description = "Timeout in seconds for socket connections", Order = 10, GroupName = "Socket Settings")]
        public int ConnectionTimeout { get; set; }
        
        [Browsable(false)]
        [XmlIgnore]
        public Series<bool> IsConnected { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Bidirectional communication indicator for RL Python model";
                Name = "RLCommunicationIndicator";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                DrawHorizontalGridLines = true;
                DrawVerticalGridLines = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;
                
                // Default settings
                ServerIP = "127.0.0.1";
                DataPort = 5000;
                OrderPort = 5001;
                WaitForFullBar = true;
                EMAShortPeriod = 9;
                EMALongPeriod = 21;
                ATRPeriod = 14;
                ADXPeriod = 14;
                MaxHistoricalBars = 6000;
                ConnectionTimeout = 5;
            }
            else if (State == State.Configure)
            {
                // Set local variables from properties
                listenIP = ServerIP;
                dataPort = DataPort;
                orderPort = OrderPort;
                waitForFullBar = WaitForFullBar;
                
                // Add the series we want to plot
                IsConnected = new Series<bool>(this);
                
                // Add technical indicators
                emaShort = EMA(EMAShortPeriod);
                emaLong = EMA(EMALongPeriod);
                atr = ATR(ATRPeriod);
                adx = ADX(ADXPeriod);
                
                // Add indicators for calculations
                AddPlot(new Stroke(System.Windows.Media.Brushes.DodgerBlue, 2), PlotStyle.Line, "ConnectionStatus");
                
                // Initialize TCP server managers
                dataServer = new TcpServerManager("Data", listenIP, dataPort, dataBufferSize, ConnectionTimeout, Print);
                orderServer = new TcpServerManager("Order", listenIP, orderPort, dataBufferSize, ConnectionTimeout, Print);
                
                // Register message handlers
                dataServer.OnMessageReceived += ProcessDataMessage;
                orderServer.OnMessageReceived += ProcessOrderMessage;
            }
            else if (State == State.DataLoaded)
            {
                // Add plot for connection status
                SetZOrder(-1);
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
            // Only process bars on the primary series
            if (BarsInProgress != 0)
                return;
            
            // Check if we have enough bars
            if (CurrentBar < 20) // Need at least 20 bars for indicators to work
                return;
            
            // Update the connection status plot
            bool isConnected = (dataServer.IsConnected && orderServer.IsConnected);
            IsConnected[0] = isConnected;
            Values[0][0] = isConnected ? 1 : 0;
            
            // If we are extracting historical data, check if we need to send more bars
            // Priorizar la extracción de datos históricos
            if (isExtractingData && !isExtractionComplete)
            {
                // Llamar a SendHistoricalBatch sin esperar a la actualización de barras
                this.Dispatcher.InvokeAsync(() => SendHistoricalBatch());
                return; // Skip regular bar sending during extraction
            }
            
            // Para barras regulares (solo si no estamos extrayendo)
            // For real-time bars, check if we should wait for the complete bar
            if (State == State.Realtime && waitForFullBar)
            {
                DateTime barTime = Time[0];
                int currentMinute = barTime.Minute;
                
                // Check if this is a new minute
                if (currentMinute != lastMinute)
                {
                    lastMinute = currentMinute;
                    
                    // Send the previous complete bar
                    if (CurrentBar > 0)
                    {
                        SendBarData(1); // Send the previous bar (index 1)
                    }
                }
            }
            else
            {
                // Send current bar data
                SendBarData(0);
            }
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
                
                // Process trade execution message (could come from data or order channel)
                if (message.StartsWith("TRADE_EXECUTED:"))
                {
                    HandleTradeExecutionMessage(message);
                    return;
                }
                
                // Handle data extraction messages
                if (message.StartsWith("EXTRACTION_") || message.Equals(extractionCommand))
                {
                    HandleExtractionMessage(message);
                    return;
                }
                
                // Log other messages
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
                
                // Print all other messages for debugging
                Print("Received order message: " + message);
                
                // Check for special command messages
                if (message.StartsWith(extractionCommand))
                {
                    // Verificar si se especificó la cantidad de barras
                    int barsToExtract = MaxHistoricalBars;  // Valor predeterminado
                    
                    // Extraer la cantidad de barras del comando si está presente
                    if (message.Contains(":"))
                    {
                        string[] parts = message.Split(':');
                        if (parts.Length > 1 && int.TryParse(parts[1], out int requestedBars))
                        {
                            barsToExtract = requestedBars;
                            Print($"Requested to extract {barsToExtract} bars");
                        }
                    }
                    
                    // Handle data extraction command
                    Print("Received data extraction command from Python. Starting extraction...");
                    this.Dispatcher.InvokeAsync(() => StartHistoricalDataExtraction(barsToExtract));
                    return;
                }
                
                // Handle data extraction-related messages
                if (message.StartsWith("EXISTING_DATA_CHECK:") || message.Equals("EXTRACT_HISTORICAL_DATA"))
                {
                    string instrumentName = "Unknown";
                    int requestedBars = MaxHistoricalBars;  // Valor predeterminado
                    
                    if (message.StartsWith("EXISTING_DATA_CHECK:"))
                    {
                        string[] dataParts = message.Split(':');
                        if (dataParts.Length > 1)
                        {
                            instrumentName = dataParts[1];
                        }
                    }
                    else if (message.StartsWith("EXTRACT_HISTORICAL_DATA:"))
                    {
                        string[] dataParts = message.Split(':');
                        if (dataParts.Length > 1 && int.TryParse(dataParts[1], out int parsedBars))
                        {
                            requestedBars = parsedBars;
                        }
                    }
                    
                    // Start historical data extraction 
                    this.Dispatcher.InvokeAsync(() => {
                        Print($"Python client requested {requestedBars} historical bars for {instrumentName}");
                        StartHistoricalDataExtraction(requestedBars);
                    });
                    
                    return;
                }
                
                // Process trade execution message (could come from data or order channel)
                if (message.StartsWith("TRADE_EXECUTED:"))
                {
                    HandleTradeExecutionMessage(message);
                    return;
                }
                
                // Process trading signals from Python
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
            
            // Call the delegate if it's assigned
            if (OnTradeSignalReceived != null)
            {
                this.Dispatcher.InvokeAsync(() => {
                    OnTradeSignalReceived(tradeSignal, emaChoice, positionSize, stopLoss, takeProfit);
                });
            }
            
            // Confirm the message was processed correctly
            orderServer.SendMessage($"ORDER_CONFIRMED:{tradeSignal},{emaChoice},{positionSize}\n");
        }
        
        private void HandleTradeExecutionMessage(string message)
        {
            try
            {
                string[] parts = message.Substring(15).Split(',');  // Remove "TRADE_EXECUTED:" prefix
                if (parts.Length >= 5)
                {
                    string action = parts[0];
                    double entryPrice = 0, exitPrice = 0, pnl = 0;
                    int quantity = 0;
                    
                    // Parse values with proper error handling
                    double.TryParse(parts[1], out entryPrice);
                    double.TryParse(parts[2], out exitPrice);
                    double.TryParse(parts[3], out pnl);
                    int.TryParse(parts[4], out quantity);
                    
                    Print($"Trade execution: {action}, P&L: {pnl}, Size: {quantity}");
                }
            }
            catch (Exception ex)
            {
                Print($"Error processing trade execution: {ex.Message}");
            }
        }
        
        private void HandleExtractionMessage(string message)
        {
            try
            {
                // Handle extraction-related messages
                if (message.StartsWith("EXTRACTION_START"))
                {
                    // Extract the total bars count and instrument name
                    string[] parts = message.Split(':');
                    if (parts.Length > 1)
                    {
                        totalBarsToExtract = int.Parse(parts[1]);
                        extractedBarsCount = 0;
                        isExtractingData = true;
                        isExtractionComplete = false;
                        
                        // Extract instrument name if available
                        string instrumentName = "Unknown";
                        if (parts.Length > 2)
                        {
                            instrumentName = parts[2].Trim();
                        }
                        
                        Print($"Starting extraction of {totalBarsToExtract} bars for {instrumentName}");
                    }
                }
                else if (message.StartsWith("CHECK_EXISTING_DATA"))
                {
                    // Respond with available data
                    string[] parts = message.Split(':');
                    if (parts.Length > 1)
                    {
                        string instrumentName = parts[1].Trim();
                        
                        // Comprobar si tenemos indicadores válidos antes de confirmar
                        bool hasValidIndicators = false;
                        for (int i = 0; i <= Math.Min(50, CurrentBar); i++)
                        {
                            if (emaShort.IsValidDataPoint(i) && emaLong.IsValidDataPoint(i) && 
                                atr.IsValidDataPoint(i) && adx.IsValidDataPoint(i))
                            {
                                hasValidIndicators = true;
                                break;
                            }
                        }
                        
                        if (hasValidIndicators)
                        {
                            Print($"Responding to CHECK_EXISTING_DATA: Indicators are valid for {instrumentName}");
                            orderServer.SendMessage($"NO_EXISTING_DATA:{instrumentName}\n");
                        }
                        else
                        {
                            Print($"Warning: No valid indicators found for {instrumentName}. Indicators may need warmup period.");
                            orderServer.SendMessage($"INDICATORS_NOT_READY:{instrumentName}\n");
                        }
                    }
                }
                else if (message.StartsWith("FORCE_EXTRACTION") || message.Equals(extractionCommand))
                {
                    // Si es FORCE_EXTRACTION, extraer ignorando validaciones de indicadores
                    bool forceExtraction = message.StartsWith("FORCE_EXTRACTION");
                    int requestedBars = 0;
                    
                    // Intentar extraer números de barras del comando
                    if (message.Contains(":"))
                    {
                        string[] parts = message.Split(':');
                        if (parts.Length > 1 && int.TryParse(parts[1], out int bars))
                        {
                            requestedBars = bars;
                        }
                    }
                    
                    // Start historical data extraction with requested bars
                    this.Dispatcher.InvokeAsync(() => StartHistoricalDataExtraction(requestedBars));
                }
            }
            catch (Exception ex)
            {
                Print($"Error handling extraction message: {ex.Message}");
            }
        }
        #endregion
        
        #region Historical Data Extraction
        public void StartHistoricalDataExtraction(int requestedBars = 0)
        {
            if (isExtractingData)
            {
                Print("Historical data extraction already in progress");
                return;
            }
            
            // Usar el valor solicitado o el valor predeterminado
            int barsRequested = requestedBars > 0 ? requestedBars : MaxHistoricalBars;
            
            // Limitar el número de barras a extraer para evitar sobrecargar NinjaTrader
            // y mantener un valor razonable que se pueda procesar
            int barsAvailable = CurrentBar + 1;
            int reasonableLimit = 6000; // Aumentado para permitir más barras
            
            // Ver cuántas barras tienen datos válidos para los indicadores
            int validIndicatorBars = 0;
            
            // Verificar desde dónde los indicadores tienen datos válidos
            for (int i = Math.Min(50, CurrentBar); i >= 0; i--)
            {
                if (emaShort.IsValidDataPoint(i) && emaLong.IsValidDataPoint(i) && 
                    atr.IsValidDataPoint(i) && adx.IsValidDataPoint(i))
                {
                    validIndicatorBars = CurrentBar - i;
                    break;
                }
            }
            
            Print($"Found {validIndicatorBars} valid bars for indicators");
            
            // Asegurarse de que no estamos tratando de extraer más barras de las que tienen indicadores válidos
            int adjustedBarsAvailable = Math.Min(barsAvailable, validIndicatorBars);
            
            // Si no hay barras válidas para los indicadores, iniciar con un valor pequeño pero válido
            if (adjustedBarsAvailable <= 0)
            {
                Print("Warning: No bars with valid indicators found. Starting with minimum value.");
                adjustedBarsAvailable = Math.Min(barsAvailable, 100); // Un valor mínimo razonable
            }
            
            // Tomar el mínimo entre el límite razonable, las barras configuradas y las disponibles
            // con indicadores válidos
            int availableBars = Math.Min(Math.Min(adjustedBarsAvailable, barsRequested), reasonableLimit);
            
            Print($"Extracting {availableBars} bars (requested: {barsRequested}, available: {barsAvailable}, with valid indicators: {adjustedBarsAvailable})");
            
            if (availableBars <= 0)
            {
                Print("Not enough bars available for extraction");
                return;
            }
            
            string instrumentName = Instrument.MasterInstrument.Name;
            
            // Establecer el estado de extracción
            isExtractingData = true;
            isExtractionComplete = false;
            totalBarsToExtract = availableBars;
            extractedBarsCount = 0;
            
            Print($"******* STARTING EXTRACTION *******");
            Print($"Starting extraction of {totalBarsToExtract} historical bars for {instrumentName}...");
            Print($"CurrentBar: {CurrentBar}, Using {availableBars} bars (limited to reasonable size)");
            
            // Enviar mensaje de inicio de extracción
            string startMessage = $"EXTRACTION_START:{totalBarsToExtract}:{instrumentName}\n";
            dataServer.SendMessage(startMessage);
            Print($"Sent extraction start message: {startMessage.Trim()}");
            
            // Procesamiento inicial en un hilo separado para no bloquear la UI
            this.Dispatcher.InvokeAsync(() => 
            {
                try
                {
                    Print("Initiating first batch of data extraction...");
                    SendHistoricalBatch();
                    
                    // Si aún no hemos terminado, programar una tarea para continuar en segundo plano
                    // con un límite máximo de tiempo para evitar bucles infinitos
                    if (isExtractingData && !isExtractionComplete)
                    {
                        DateTime startTime = DateTime.Now;
                        TimeSpan maxExtractionTime = TimeSpan.FromMinutes(2); // Limitamos a 2 minutos máximo
                        
                        this.Dispatcher.InvokeAsync(() => 
                        {
                            int previousProgress = 0;
                            int noProgressCount = 0;
                            int maxNoProgressAttempts = 5;
                            int retryCount = 0;
                            int maxRetries = 3;
                            int currentBatchSize = barsRequested;
                            
                            // Continuar enviando lotes hasta completar o alcanzar el límite de tiempo
                            while (isExtractingData && !isExtractionComplete)
                            {
                                // Verificar si hemos excedido el tiempo máximo
                                if (DateTime.Now - startTime > maxExtractionTime)
                                {
                                    Print($"Extraction time limit exceeded ({maxExtractionTime.TotalMinutes} minutes). Stopping.");
                                    isExtractionComplete = true;
                                    isExtractingData = false;
                                    dataServer.SendMessage($"EXTRACTION_COMPLETE:{instrumentName}\n");
                                    break;
                                }
                                
                                // Enviar un lote de datos
                                SendHistoricalBatch();
                                
                                // Verificar si estamos haciendo progreso
                                if (extractedBarsCount == previousProgress)
                                {
                                    noProgressCount++;
                                    Print($"No progress made in extraction. Attempt {noProgressCount}/{maxNoProgressAttempts}");
                                    
                                    if (noProgressCount >= maxNoProgressAttempts)
                                    {
                                        // Intentar reducir el tamaño del lote si alcanzamos el límite de intentos sin progreso
                                        if (retryCount < maxRetries)
                                        {
                                            retryCount++;
                                            // Reducir el número de barras en cada intento
                                            currentBatchSize = currentBatchSize / 2;
                                            if (currentBatchSize < 100) currentBatchSize = 100; // Mínimo de 100 barras
                                            
                                            // Ajustar el total para el nuevo tamaño
                                            totalBarsToExtract = Math.Min(currentBatchSize, totalBarsToExtract);
                                            
                                            Print($"Retrying with reduced batch size: {currentBatchSize} bars. Retry {retryCount}/{maxRetries}");
                                            
                                            // Reiniciar contadores
                                            noProgressCount = 0;
                                            continue;
                                        }
                                        
                                        Print("Maximum attempts with no progress reached. Stopping extraction.");
                                        isExtractionComplete = true;
                                        isExtractingData = false;
                                        dataServer.SendMessage($"EXTRACTION_COMPLETE:{instrumentName}\n");
                                        break;
                                    }
                                }
                                else
                                {
                                    // Reiniciar contador si hicimos progreso
                                    noProgressCount = 0;
                                    previousProgress = extractedBarsCount;
                                }
                                
                                // Pequeña pausa para no saturar el sistema
                                System.Threading.Thread.Sleep(100);
                            }
                            
                            Print($"Background extraction process finished. Extracted {extractedBarsCount} bars.");
                        });
                    }
                }
                catch (Exception ex)
                {
                    Print($"Error during data extraction: {ex.Message}");
                    isExtractionComplete = true;
                    isExtractingData = false;
                    dataServer.SendMessage($"EXTRACTION_COMPLETE:{instrumentName}\n");
                }
            });
        }
        
        private void SendHistoricalBatch()
        {
            // Verificar si ya completamos la extracción
            if (extractedBarsCount >= totalBarsToExtract || !isExtractingData)
            {
                if (!isExtractionComplete)
                {
                    string instrumentName = Instrument.MasterInstrument.Name;
                    string completeMessage = $"EXTRACTION_COMPLETE:{instrumentName}\n";
                    dataServer.SendMessage(completeMessage);
                    
                    isExtractionComplete = true;
                    isExtractingData = false;
                    Print($"Historical data extraction completed for {instrumentName}: {extractedBarsCount} bars sent");
                }
                return;
            }
            
            int batchSize = 10; // Reducido aún más para evitar sobrecarga y problemas
            int remainingBars = totalBarsToExtract - extractedBarsCount;
            int barsToSend = Math.Min(batchSize, remainingBars);
            int sentInThisBatch = 0;
            int errorCount = 0;
            int maxErrorsAllowed = 10;
            
            // Verificar que tenemos suficientes barras disponibles
            if (CurrentBar < barsToSend)
            {
                Print($"Not enough bars available. CurrentBar: {CurrentBar}, barsToSend: {barsToSend}");
                
                // Finalizar la extracción si no hay suficientes barras
                isExtractionComplete = true;
                isExtractingData = false;
                
                string instrumentName = Instrument.MasterInstrument.Name;
                string completeMessage = $"EXTRACTION_COMPLETE:{instrumentName}\n";
                dataServer.SendMessage(completeMessage);
                
                Print($"Extraction terminated - insufficient data. Got {extractedBarsCount} of {totalBarsToExtract} requested bars.");
                return;
            }
            
            // Verificar el estado de los indicadores para esta ventana de tiempo
            bool hasValidIndicators = false;
            for (int i = 0; i < barsToSend; i++)
            {
                int barIndex = extractedBarsCount + i;
                int barsAgo = CurrentBar - barIndex;
                
                if (barsAgo >= 0 && barsAgo <= CurrentBar)
                {
                    if (emaShort.IsValidDataPoint(barsAgo) && emaLong.IsValidDataPoint(barsAgo) && 
                        atr.IsValidDataPoint(barsAgo) && adx.IsValidDataPoint(barsAgo))
                    {
                        hasValidIndicators = true;
                        break;
                    }
                }
            }
            
            if (!hasValidIndicators)
            {
                Print($"No valid indicator data in this batch range: {extractedBarsCount} to {extractedBarsCount + barsToSend - 1}");
                
                // Avanzar al siguiente lote pero sin incrementar el contador de errores
                extractedBarsCount += barsToSend;
                
                // Enviar mensaje de progreso
                double currentProgressPercent = totalBarsToExtract > 0 ? (double)extractedBarsCount / totalBarsToExtract * 100 : 0;
                string currentProgressMessage = $"EXTRACTION_PROGRESS:{extractedBarsCount}:{totalBarsToExtract}:{currentProgressPercent:F1}\n";
                dataServer.SendMessage(currentProgressMessage);
                
                Print($"Skipped batch due to invalid indicators. Moving to next batch: {extractedBarsCount}");
                
                return;
            }
            
            Print($"Sending batch: from={extractedBarsCount} to {extractedBarsCount + barsToSend - 1}, CurrentBar={CurrentBar}");
            
            // Comenzar desde la barra más antigua y avanzar hacia las más recientes
            // Invertir el orden de envío para evitar errores de índice
            for (int i = 0; i < barsToSend && errorCount < maxErrorsAllowed; i++)
            {
                // Calcular el índice de barra en el rango [0, CurrentBar]
                // barsAgo indica cuántas barras ATRÁS desde la barra actual, por lo que
                // 0 = la barra actual, 1 = la barra anterior, etc.
                int barIndex = extractedBarsCount + i;
                
                // El valor de barsAgo es CurrentBar - el índice de la barra que queremos
                // IMPORTANTE: Si barIndex es mayor que CurrentBar, esto puede resultar en valores negativos
                int barsAgo = CurrentBar - barIndex;
                
                // Validación más estricta para evitar índices fuera de rango
                if (barsAgo >= 0 && barsAgo <= CurrentBar && barIndex >= 0 && barIndex <= totalBarsToExtract)
                {
                    try
                    {
                        // Usar el método común para formatear los datos
                        string message = FormatBarData(barsAgo, true);
                        if (message != null)
                        {
                            if (i == 0 || i % 20 == 0) // Reducir la verbosidad del log
                            {
                                Print($"Sending bar: barsAgo={barsAgo}, barIndex={barIndex}, time={Time[barsAgo].ToString()}");
                            }
                            
                            dataServer.SendMessage(message);
                            sentInThisBatch++;
                        }
                    }
                    catch (Exception ex)
                    {
                        // Contar errores pero continuar con la siguiente barra
                        errorCount++;
                        Print($"Error sending bar with barsAgo={barsAgo}: {ex.Message}");
                        
                        if (errorCount >= maxErrorsAllowed)
                        {
                            Print($"Too many errors ({errorCount}). Stopping batch processing.");
                            break;
                        }
                    }
                }
                else
                {
                    // Este es un error crítico, contar y registrar
                    errorCount++;
                    Print($"Invalid barsAgo={barsAgo} (barIndex={barIndex}, CurrentBar={CurrentBar}) - outside valid range [0, {CurrentBar}]");
                    
                    if (errorCount >= maxErrorsAllowed)
                    {
                        Print($"Too many errors ({errorCount}). Stopping batch processing.");
                        break;
                    }
                }
            }
            
            // Actualizar el contador de barras extraídas
            extractedBarsCount += sentInThisBatch;
            
            // Enviar mensaje de progreso
            double progressPercent = totalBarsToExtract > 0 ? (double)extractedBarsCount / totalBarsToExtract * 100 : 0;
            string progressMessage = $"EXTRACTION_PROGRESS:{extractedBarsCount}:{totalBarsToExtract}:{progressPercent:F1}\n";
            dataServer.SendMessage(progressMessage);
            
            // Verificar si hemos completado la extracción o si tuvimos demasiados errores para continuar
            if (extractedBarsCount >= totalBarsToExtract || (sentInThisBatch == 0 && errorCount > 0))
            {
                string instrumentName = Instrument.MasterInstrument.Name;
                string completeMessage = $"EXTRACTION_COMPLETE:{instrumentName}\n";
                dataServer.SendMessage(completeMessage);
                
                isExtractionComplete = true;
                isExtractingData = false;
                
                if (sentInThisBatch == 0 && errorCount > 0)
                {
                    Print($"Extraction terminated early due to errors. Got {extractedBarsCount} of {totalBarsToExtract} requested bars.");
                }
                else
                {
                    Print($"Historical data extraction completed: {extractedBarsCount} bars sent");
                }
            }
            
            Print($"Extraction progress: {progressPercent:F1}% ({extractedBarsCount}/{totalBarsToExtract}, sent {sentInThisBatch} bars in this batch)");
        }
        
        private string FormatBarData(int barsAgo, bool isHistorical = false)
        {
            // Format a bar's data into a string
            try
            {
                // Verificación exhaustiva del índice barsAgo
                if (barsAgo < 0)
                {
                    Print($"Invalid barsAgo: {barsAgo} (negative value)");
                    return null;
                }
                
                if (barsAgo > CurrentBar)
                {
                    Print($"Invalid barsAgo: {barsAgo} (exceeds CurrentBar: {CurrentBar})");
                    return null;
                }
                
                // Comprobar y registrar el estado de los indicadores individualmente
                bool emaShortValid = emaShort.IsValidDataPoint(barsAgo);
                bool emaLongValid = emaLong.IsValidDataPoint(barsAgo);
                bool atrValid = atr.IsValidDataPoint(barsAgo);
                bool adxValid = adx.IsValidDataPoint(barsAgo);
                
                if (!emaShortValid) Print($"EMA Short not valid at barsAgo={barsAgo}, CurrentBar={CurrentBar}");
                if (!emaLongValid) Print($"EMA Long not valid at barsAgo={barsAgo}, CurrentBar={CurrentBar}");
                if (!atrValid) Print($"ATR not valid at barsAgo={barsAgo}, CurrentBar={CurrentBar}");
                if (!adxValid) Print($"ADX not valid at barsAgo={barsAgo}, CurrentBar={CurrentBar}");
                
                // Si alguno no es válido pero estamos cerca del final, usar valores predeterminados
                if (!emaShortValid || !emaLongValid || !atrValid || !adxValid)
                {
                    // Si estamos cerca del final (últimos 30 períodos), usar valores predeterminados
                    if (barsAgo <= 30 || CurrentBar - barsAgo <= 30)
                    {
                        Print($"Using default values for indicators at barsAgo={barsAgo} (near edge of dataset)");
                        
                        // Obtener el nombre del instrumento
                        string currentInstrumentName = Instrument.MasterInstrument.Name;
                        
                        // Create formatter string con el nombre del instrumento agregado
                        string currentPrefix = isHistorical ? "HISTORICAL:" : "";
                        
                        // Usar valores actuales para los cálculos válidos, valores predeterminados para los inválidos
                        double emaShortValue = emaShortValid ? emaShort[barsAgo] : Close[barsAgo];
                        double emaLongValue = emaLongValid ? emaLong[barsAgo] : Close[barsAgo];
                        double atrValue = atrValid ? atr[barsAgo] : 0.01;
                        double adxValue = adxValid ? adx[barsAgo] : 25;
                        
                        string formattedBarData = string.Format("{0}{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}",
                            currentPrefix,
                            currentInstrumentName,  // Agregamos el nombre del instrumento como primer campo
                            Open[barsAgo].ToString("F2"),
                            High[barsAgo].ToString("F2"),
                            Low[barsAgo].ToString("F2"),
                            Close[barsAgo].ToString("F2"),
                            emaShortValue.ToString("F2"),
                            emaLongValue.ToString("F2"),
                            atrValue.ToString("F2"),
                            adxValue.ToString("F2"),
                            Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"),
                            Time[barsAgo].ToOADate()); // Agregar valor numérico de fecha para facilitar ordenamiento
                        
                        // Add newline for message separation
                        return formattedBarData + "\n";
                    }
                    else
                    {
                        Print($"Invalid data point for indicators at barsAgo={barsAgo}");
                        return null;
                    }
                }
                
                // Si todos los indicadores son válidos
                // Obtener el nombre del instrumento
                string validInstrumentName = Instrument.MasterInstrument.Name;
                
                // Create formatter string con el nombre del instrumento agregado
                string validPrefix = isHistorical ? "HISTORICAL:" : "";
                string validBarData = string.Format("{0}{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}",
                    validPrefix,
                    validInstrumentName,  // Agregamos el nombre del instrumento como primer campo
                    Open[barsAgo].ToString("F2"),
                    High[barsAgo].ToString("F2"),
                    Low[barsAgo].ToString("F2"),
                    Close[barsAgo].ToString("F2"),
                    emaShort[barsAgo].ToString("F2"),
                    emaLong[barsAgo].ToString("F2"),
                    atr[barsAgo].ToString("F2"),
                    adx[barsAgo].ToString("F2"),
                    Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"),
                    Time[barsAgo].ToOADate()); // Agregar valor numérico de fecha para facilitar ordenamiento
                
                // Add newline for message separation
                return validBarData + "\n";
            }
            catch (Exception ex)
            {
                Print($"Error formatting bar data for barsAgo={barsAgo}: {ex.Message}");
                return null;
            }
        }
        
        private void SendBarData(int barIndex)
        {
            if (barIndex < 0 || barIndex > CurrentBar)
            {
                Print($"Invalid bar index for SendBarData: {barIndex}, CurrentBar: {CurrentBar}");
                return;
            }
                
            try
            {
                // En NinjaTrader, debemos utilizar BarsAgo, que es la DIFERENCIA entre CurrentBar y el índice deseado
                int barsAgo = CurrentBar - barIndex;
                
                // Format and send the data
                string message = FormatBarData(barsAgo, false);
                if (message != null)
                {
                    dataServer.SendMessage(message);
                }
            }
            catch (Exception ex)
            {
                Print($"Error sending data for index {barIndex}: {ex.Message}");
            }
        }
        #endregion

        #region Public Methods
        // Public method to send trade execution data back to Python
        public void SendTradeExecutionUpdate(string action, double entryPrice, double exitPrice, double pnl, int quantity)
        {
            try
            {
                string executionMessage = $"TRADE_EXECUTED:{action},{entryPrice},{exitPrice},{pnl},{quantity}\n";
                orderServer.SendMessage(executionMessage);
                Print($"Sent trade execution data: {action}, P&L: {pnl}");
            }
            catch (Exception ex)
            {
                Print($"Error sending trade execution update: {ex.Message}");
            }
        }
        #endregion
        
        public override string ToString()
        {
            bool dataConnected = dataServer != null && dataServer.IsConnected;
            bool orderConnected = orderServer != null && orderServer.IsConnected;
            string status = " [";
            
            if (dataConnected && orderConnected)
                status += "Fully Connected";
            else if (dataConnected)
                status += "Data Connected";
            else if (orderConnected)
                status += "Order Connected";
            else
                status += "Disconnected";
                
            status += "]";
            
            return Name + status;
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private RLCommunicationIndicator[] cacheRLCommunicationIndicator;
		public RLCommunicationIndicator RLCommunicationIndicator(string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			return RLCommunicationIndicator(Input, serverIP, dataPort, orderPort, waitForFullBar, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, maxHistoricalBars, connectionTimeout);
		}

		public RLCommunicationIndicator RLCommunicationIndicator(ISeries<double> input, string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			if (cacheRLCommunicationIndicator != null)
				for (int idx = 0; idx < cacheRLCommunicationIndicator.Length; idx++)
					if (cacheRLCommunicationIndicator[idx] != null && cacheRLCommunicationIndicator[idx].ServerIP == serverIP && cacheRLCommunicationIndicator[idx].DataPort == dataPort && cacheRLCommunicationIndicator[idx].OrderPort == orderPort && cacheRLCommunicationIndicator[idx].WaitForFullBar == waitForFullBar && cacheRLCommunicationIndicator[idx].EMAShortPeriod == eMAShortPeriod && cacheRLCommunicationIndicator[idx].EMALongPeriod == eMALongPeriod && cacheRLCommunicationIndicator[idx].ATRPeriod == aTRPeriod && cacheRLCommunicationIndicator[idx].ADXPeriod == aDXPeriod && cacheRLCommunicationIndicator[idx].MaxHistoricalBars == maxHistoricalBars && cacheRLCommunicationIndicator[idx].ConnectionTimeout == connectionTimeout && cacheRLCommunicationIndicator[idx].EqualsInput(input))
						return cacheRLCommunicationIndicator[idx];
			return CacheIndicator<RLCommunicationIndicator>(new RLCommunicationIndicator(){ ServerIP = serverIP, DataPort = dataPort, OrderPort = orderPort, WaitForFullBar = waitForFullBar, EMAShortPeriod = eMAShortPeriod, EMALongPeriod = eMALongPeriod, ATRPeriod = aTRPeriod, ADXPeriod = aDXPeriod, MaxHistoricalBars = maxHistoricalBars, ConnectionTimeout = connectionTimeout }, input, ref cacheRLCommunicationIndicator);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.RLCommunicationIndicator RLCommunicationIndicator(string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			return indicator.RLCommunicationIndicator(Input, serverIP, dataPort, orderPort, waitForFullBar, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, maxHistoricalBars, connectionTimeout);
		}

		public Indicators.RLCommunicationIndicator RLCommunicationIndicator(ISeries<double> input , string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			return indicator.RLCommunicationIndicator(input, serverIP, dataPort, orderPort, waitForFullBar, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, maxHistoricalBars, connectionTimeout);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.RLCommunicationIndicator RLCommunicationIndicator(string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			return indicator.RLCommunicationIndicator(Input, serverIP, dataPort, orderPort, waitForFullBar, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, maxHistoricalBars, connectionTimeout);
		}

		public Indicators.RLCommunicationIndicator RLCommunicationIndicator(ISeries<double> input , string serverIP, int dataPort, int orderPort, bool waitForFullBar, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int maxHistoricalBars, int connectionTimeout)
		{
			return indicator.RLCommunicationIndicator(input, serverIP, dataPort, orderPort, waitForFullBar, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, maxHistoricalBars, connectionTimeout);
		}
	}
}

#endregion
