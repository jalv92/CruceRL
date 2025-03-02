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
        #region Variables
        // Server components
        private TcpListener dataServer;
        private TcpListener orderServer;
        private TcpClient dataClient;
        private TcpClient orderClient;
        private NetworkStream dataClientStream;
        private NetworkStream orderClientStream;
        private string listenIP = "127.0.0.1";
        private int dataPort = 5000;
        private int orderPort = 5001;
        private DateTime lastConnectionAttempt = DateTime.MinValue;
        private bool isConnecting = false;
        private int reconnectInterval = 5; // seconds
        private int dataBufferSize = 60; // Number of bars to buffer
        private Queue<string> dataBuffer = new Queue<string>();
        private bool isInitialDataSent = false;
        private Thread dataServerThread;
        private Thread orderServerThread;
        private bool isDataServerRunning = false;
        private bool isOrderServerRunning = false;
        private int lastMinute = -1;
        private bool waitForFullBar = true;
        private bool connectionErrorLogged = false; // Prevent log spam
        
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
                    
                    bool dataServerStarted = StartDataServer();
                    Thread.Sleep(500);
                    bool orderServerStarted = StartOrderServer();
                    
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
                StopDataServer();
                StopOrderServer();
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
            bool isConnected = (dataClient != null && dataClient.Connected && 
                               orderClient != null && orderClient.Connected);
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

        #region Data Server Methods
        private bool StartDataServer()
        {
            if (isDataServerRunning)
                return true;
                
            try
            {
                // Close any existing connections first
                StopDataServer();
                
                // Set up a TCP listener for the data port
                // NinjaTrader now acts as server, listening for Python client
                Print($"Starting data server on {listenIP}:{dataPort}...");
                
                // Display local IP for debugging
                try {
                    IPHostEntry host = Dns.GetHostEntry(Dns.GetHostName());
                    foreach(IPAddress ip in host.AddressList) {
                        Print($"Available local IP: {ip.ToString()}");
                    }
                } catch (Exception ex) {
                    Print($"Error getting local IPs: {ex.Message}");
                }
                
                // Create server on loopback address
                IPAddress ipAddress = IPAddress.Parse(listenIP);
                dataServer = new TcpListener(ipAddress, dataPort);
                
                // Start listening
                dataServer.Start();
                Print($"Data server started on {listenIP}:{dataPort}");
                
                // Start the server thread to accept connections
                isDataServerRunning = true;
                dataServerThread = new Thread(DataServerLoop);
                dataServerThread.IsBackground = true;
                dataServerThread.Start();
                
                return true;
            }
            catch (Exception ex)
            {
                Print($"Error starting data server: {ex.Message}");
                
                // Cleanup in case of failure
                if (dataServer != null)
                {
                    try { dataServer.Stop(); } catch { }
                    dataServer = null;
                }
                
                isDataServerRunning = false;
                return false;
            }
        }
        
        private void DataServerLoop()
        {
            Print("Data server thread started. Waiting for Python client connection...");
            
            while (isDataServerRunning)
            {
                try
                {
                    // Check for pending connections
                    if (dataServer.Pending())
                    {
                        // Accept the client connection
                        dataClient = dataServer.AcceptTcpClient();
                        dataClient.NoDelay = true;
                        dataClient.SendBufferSize = 65536;
                        dataClient.ReceiveBufferSize = 65536;
                        
                        // Get the client stream
                        dataClientStream = dataClient.GetStream();
                        
                        Print($"Python client connected to data server from {dataClient.Client.RemoteEndPoint}");
                        
                        // Send welcome message
                        SendDataMessage("SERVER_READY\n");
                        
                        // Process data from the client
                        byte[] buffer = new byte[4096];
                        StringBuilder messageBuilder = new StringBuilder();
                        
                        // Send accumulated data from buffer
                        while (dataBuffer.Count > 0)
                        {
                            try
                            {
                                string data = dataBuffer.Dequeue();
                                SendDataMessage(data);
                            }
                            catch (Exception ex)
                            {
                                Print($"Error sending buffered data: {ex.Message}");
                            }
                        }
                        
                        // Main client data reception loop
                        while (isDataServerRunning && dataClient != null && dataClient.Connected)
                        {
                            // Check if there's data to read
                            if (dataClient.Available > 0)
                            {
                                int bytesRead = dataClientStream.Read(buffer, 0, buffer.Length);
                                
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
                                        ProcessDataClientMessage(completeMessage);
                                    }
                                }
                                
                                // Update buffer with remaining incomplete message
                                messageBuilder.Clear();
                                messageBuilder.Append(message);
                            }
                            
                            // Short sleep to prevent high CPU usage
                            Thread.Sleep(10);
                            
                            // Send heartbeat periodically
                            if (DateTime.Now - lastConnectionAttempt > TimeSpan.FromSeconds(5))
                            {
                                try
                                {
                                    SendDataMessage("PING\n");
                                    lastConnectionAttempt = DateTime.Now;
                                }
                                catch
                                {
                                    // Connection might be broken
                                    break;
                                }
                            }
                        }
                        
                        // Client disconnected or error occurred
                        Print("Python client disconnected from data server");
                        
                        // Clean up client resources
                        if (dataClientStream != null)
                        {
                            try { dataClientStream.Close(); } catch { }
                            dataClientStream = null;
                        }
                        
                        if (dataClient != null)
                        {
                            try { dataClient.Close(); } catch { }
                            dataClient = null;
                        }
                    }
                    
                    // Short sleep before checking for new connections
                    Thread.Sleep(100);
                }
                catch (Exception ex)
                {
                    Print($"Error in data server loop: {ex.Message}");
                    
                    // Clean up client resources
                    if (dataClientStream != null)
                    {
                        try { dataClientStream.Close(); } catch { }
                        dataClientStream = null;
                    }
                    
                    if (dataClient != null)
                    {
                        try { dataClient.Close(); } catch { }
                        dataClient = null;
                    }
                    
                    // Short delay before retrying
                    Thread.Sleep(1000);
                }
            }
            
            Print("Data server thread stopped");
        }
        
        private void ProcessDataClientMessage(string message)
        {
            try
            {
                // Handle incoming messages from Python client
                if (message.Equals("PONG", StringComparison.OrdinalIgnoreCase))
                {
                    // Heartbeat response
                    lastConnectionAttempt = DateTime.Now;
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
                    SendDataMessage($"SERVER_INFO:NinjaTrader:{Instrument.MasterInstrument.Name}\n");
                    return;
                }
                
                // Log other messages
                Print($"Received from Python client: {message}");
            }
            catch (Exception ex)
            {
                Print($"Error processing data client message: {ex.Message}");
            }
        }

        private void StopDataServer()
        {
            try
            {
                // First mark thread as stopping
                isDataServerRunning = false;
                
                // Close client streams and connections
                if (dataClientStream != null)
                {
                    try 
                    {
                        dataClientStream.Flush();
                        dataClientStream.Close();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error closing data client stream: {ex.Message}");
                    }
                    finally
                    {
                        dataClientStream = null;
                    }
                }
                
                if (dataClient != null)
                {
                    try
                    {
                        dataClient.Close();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error closing data client: {ex.Message}");
                    }
                    finally
                    {
                        dataClient = null;
                    }
                }
                
                // Stop the server
                if (dataServer != null)
                {
                    try
                    {
                        dataServer.Stop();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error stopping data server: {ex.Message}");
                    }
                    finally
                    {
                        dataServer = null;
                    }
                }
                
                // Wait for thread to terminate
                if (dataServerThread != null && dataServerThread.IsAlive)
                {
                    try
                    {
                        dataServerThread.Join(2000);
                        if (dataServerThread.IsAlive)
                        {
                            // Not ideal but needed in case thread is stuck
                            dataServerThread.Abort();
                        }
                    }
                    catch { }
                    dataServerThread = null;
                }
                
                Print("Data server stopped successfully");
            }
            catch (Exception ex)
            {
                Print($"Error stopping data server: {ex.Message}");
            }
            finally
            {
                // Force resource cleanup
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // Method for extracting historical data on demand
        public void StartHistoricalDataExtraction()
        {
            if (isExtractingData)
            {
                Print("Historical data extraction already in progress");
                return;
            }
            
            // Limitar a un número razonable (5000 barras)
            int reasonableLimit = 5000;
            
            // Asegurar que no se intenten extraer más barras de las que están disponibles
            // o más de las que podemos procesar razonablemente
            int availableBars = Math.Min(Math.Min(CurrentBar, MaxHistoricalBars), reasonableLimit);
            
            if (availableBars <= 0)
            {
                Print("Not enough bars available for extraction");
                return;
            }
            
            // Get current instrument name
            string instrumentName = Instrument.MasterInstrument.Name;
            
            // Initialize extraction variables
            isExtractingData = true;
            isExtractionComplete = false;
            totalBarsToExtract = availableBars;
            extractedBarsCount = 0;
            
            Print($"******* STARTING EXTRACTION *******");
            Print($"Starting extraction of {totalBarsToExtract} historical bars for {instrumentName}...");
            Print($"CurrentBar: {CurrentBar}, Using {availableBars} bars (limited to reasonable size)");
            
            // Send extraction start notification to Python
            string startMessage = $"EXTRACTION_START:{totalBarsToExtract}:{instrumentName}\n";
            SendDataMessage(startMessage);
            Print($"Sent extraction start message: {startMessage.Trim()}");
            
            // Start the extraction process
            // This will automatically trigger the SendHistoricalBatch method
            // in OnBarUpdate
            
            // Send first batch immediately
            this.Dispatcher.InvokeAsync(() => 
            {
                Print("Initiating first batch of data extraction...");
                SendHistoricalBatch();
                Print($"Initial batch of historical data sent. Progress: {(double)extractedBarsCount / totalBarsToExtract * 100:F1}%");
                
                // Programar la siguiente extracción de lote inmediatamente después
                this.Dispatcher.InvokeAsync(() => 
                {
                    while (isExtractingData && !isExtractionComplete)
                    {
                        SendHistoricalBatch();
                        System.Threading.Thread.Sleep(100); // Pequeña pausa para evitar sobrecargar el sistema
                    }
                });
            });
        }
        
        // Variable to store the date from which to filter data
        private DateTime? startFromDate = null;
        
        private void SendHistoricalBatch()
        {
            // Error con "barsAgo needed to be between 0 and 29999" indica problema en cómo accedemos
            // a los datos históricos. Arreglemos eso.
            
            // Send batches of 100 bars at a time to avoid overloading the connection
            int batchSize = 100;
            int barsToSend = Math.Min(batchSize, totalBarsToExtract - extractedBarsCount);
            
            if (barsToSend <= 0)
            {
                // Extraction complete
                if (!isExtractionComplete)
                {
                    // Include instrument name in the completion message
                    string instrumentName = Instrument.MasterInstrument.Name;
                    string completeMessage = $"EXTRACTION_COMPLETE:{instrumentName}\n";
                    SendDataMessage(completeMessage);
                    
                    isExtractionComplete = true;
                    isExtractingData = false;
                    Print($"Historical data extraction completed for {instrumentName}: {extractedBarsCount} bars sent");
                    
                    // Reset start date after completion
                    startFromDate = null;
                }
                return;
            }
            
            // Send a batch of historical bars
            int sentInThisBatch = 0;
            
            try
            {
                // ENFOQUE SIMPLIFICADO: 
                // En lugar de usar índices basados en CurrentBar, usamos directamente BarsAgo
                // que es lo que NinjaTrader espera
                
                // Queremos extraer las últimas X barras, empezando por las más recientes
                // Entonces necesitamos partir de BarsAgo=0 (la barra actual)
                // hasta BarsAgo=totalBarsToExtract-1 (la barra más antigua que queremos)
                
                // Calcular el rango de BarsAgo para este lote
                int startBarsAgo = extractedBarsCount;
                int endBarsAgo = Math.Min(startBarsAgo + barsToSend - 1, totalBarsToExtract - 1);
                
                Print($"Sending batch: startBarsAgo={startBarsAgo}, endBarsAgo={endBarsAgo}");
                
                // Extraer y enviar cada barra en el rango
                // FIXED: Usar recorrido de barras basado en índices directos, no en barsAgo
                // Calculamos la posición real en el buffer en términos de índice, no de barsAgo
                for (int i = 0; i < barsToSend; i++)
                {
                    // Calcular el índice real en el buffer
                    int barIndex = extractedBarsCount + i;
                    
                    // Verificar que el barIndex está dentro del rango permitido
                    if (barIndex < CurrentBar)
                    {
                        try
                        {
                            // Convertir el índice a barsAgo para acceder a los datos
                            int barsAgoValue = CurrentBar - barIndex;
                            
                            // Preparar mensaje usando el valor correcto de barsAgo
                            string message = string.Format("HISTORICAL:{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                                Open[barsAgoValue].ToString("F2"),
                                High[barsAgoValue].ToString("F2"),
                                Low[barsAgoValue].ToString("F2"),
                                Close[barsAgoValue].ToString("F2"),
                                emaShort[barsAgoValue].ToString("F2"),
                                emaLong[barsAgoValue].ToString("F2"),
                                atr[barsAgoValue].ToString("F4"),
                                adx[barsAgoValue].ToString("F2"),
                                Time[barsAgoValue].ToString("yyyy-MM-dd HH:mm:ss"));
                            
                            // Add newline for message separation
                            message += "\n";
                            
                            // Log occasionally for debugging
                            if (sentInThisBatch % 20 == 0)
                            {
                                Print($"Sending bar: barIndex={barIndex}, barsAgo={barsAgoValue}, time={Time[barsAgoValue].ToString()}");
                            }
                            
                            // Send the message
                            SendDataMessage(message);
                            sentInThisBatch++;
                        }
                        catch (Exception ex)
                        {
                            Print($"Error sending bar {barIndex}: {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"Error sending batch: {ex.Message}");
            }
            
            // Update the count of extracted bars
            extractedBarsCount += sentInThisBatch;
            
            // Send progress update
            double progressPercent = totalBarsToExtract > 0 ? (double)extractedBarsCount / totalBarsToExtract * 100 : 0;
            string progressMessage = $"EXTRACTION_PROGRESS:{extractedBarsCount}:{totalBarsToExtract}:{progressPercent:F1}\n";
            SendDataMessage(progressMessage);
            
            // Verificar si hemos completado la extracción
            if (extractedBarsCount >= totalBarsToExtract)
            {
                string instrumentName = Instrument.MasterInstrument.Name;
                string completeMessage = $"EXTRACTION_COMPLETE:{instrumentName}\n";
                SendDataMessage(completeMessage);
                
                isExtractionComplete = true;
                isExtractingData = false;
                Print($"Historical data extraction completed for {instrumentName}: {extractedBarsCount} bars sent");
            }
            
            Print($"Extraction progress: {progressPercent:F1}% ({extractedBarsCount}/{totalBarsToExtract}, sent {sentInThisBatch} bars in this batch)");
        }
        
        private void SendBarDataForExtraction(int barIndex)
        {
            if (barIndex < 0 || barIndex >= CurrentBar)
            {
                Print($"Invalid bar index: {barIndex}, CurrentBar: {CurrentBar}");
                return;
            }
                
            try
            {
                // En NinjaTrader, debemos utilizar BarsAgo, que es la DIFERENCIA entre CurrentBar y el índice deseado
                // Esto quita el error "barsAgo needed to be between 0 and 29999"
                int barsAgo = CurrentBar - barIndex;
                
                // Prepare data message with OHLC and indicators
                string message = string.Format("HISTORICAL:{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                    Open[barsAgo].ToString("F2"),
                    High[barsAgo].ToString("F2"),
                    Low[barsAgo].ToString("F2"),
                    Close[barsAgo].ToString("F2"),
                    emaShort[barsAgo].ToString("F2"),
                    emaLong[barsAgo].ToString("F2"),
                    atr[barsAgo].ToString("F4"),
                    adx[barsAgo].ToString("F2"),
                    Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"));
                
                // Add newline for message separation
                message += "\n";
                
                // Log occasionally for debugging (every 100 bars)
                if (extractedBarsCount % 100 == 0)
                {
                    Print($"Sending historical bar data: barIndex={barIndex}, barsAgo={barsAgo}, message={message.Trim()}");
                }
                
                // Send the message
                SendDataMessage(message);
            }
            catch (Exception ex)
            {
                Print($"Error sending historical bar data for index {barIndex}: {ex.Message}");
            }
        }

        private void SendBarData(int barIndex)
        {
            if (barIndex < 0 || barIndex >= CurrentBar)
            {
                Print($"Invalid bar index for SendBarData: {barIndex}, CurrentBar: {CurrentBar}");
                return;
            }
                
            try
            {
                // En NinjaTrader, debemos utilizar BarsAgo, que es la DIFERENCIA entre CurrentBar y el índice deseado
                int barsAgo = CurrentBar - barIndex;
                
                // Prepare data message with OHLC and indicators
                string message = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                    Open[barsAgo].ToString("F2"),
                    High[barsAgo].ToString("F2"),
                    Low[barsAgo].ToString("F2"),
                    Close[barsAgo].ToString("F2"),
                    emaShort[barsAgo].ToString("F2"),
                    emaLong[barsAgo].ToString("F2"),
                    atr[barsAgo].ToString("F4"),
                    adx[barsAgo].ToString("F2"),
                    Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"));
                
                // Add newline for message separation
                message += "\n";
                
                // Send the message
                SendDataMessage(message);
            }
            catch (Exception ex)
            {
                Print($"Error sending data for index {barIndex}: {ex.Message}");
            }
        }
        
        private void SendDataMessage(string message)
        {
            try
            {
                // Check if connected to a client
                if (dataClient != null && dataClient.Connected && dataClientStream != null)
                {
                    // Send data directly
                    byte[] data = Encoding.ASCII.GetBytes(message);
                    dataClientStream.Write(data, 0, data.Length);
                }
                else
                {
                    // Buffer the data for when a client connects
                    dataBuffer.Enqueue(message);
                    
                    // Limit buffer size
                    while (dataBuffer.Count > dataBufferSize)
                    {
                        dataBuffer.Dequeue();
                    }
                }
            }
            catch (Exception ex)
            {
                Print("Error sending data message: " + ex.Message);
                
                // Buffer the message in case client reconnects
                dataBuffer.Enqueue(message);
                
                // Limit buffer size
                while (dataBuffer.Count > dataBufferSize)
                {
                    dataBuffer.Dequeue();
                }
                
                // Client connection might be broken, clean up
                if (dataClientStream != null)
                {
                    try { dataClientStream.Close(); } catch { }
                    dataClientStream = null;
                }
                
                if (dataClient != null)
                {
                    try { dataClient.Close(); } catch { }
                    dataClient = null;
                }
            }
        }
        #endregion

        #region Order Server Methods
        private bool StartOrderServer()
        {
            if (isOrderServerRunning)
                return true;
                
            try
            {
                // Close any existing connections first
                StopOrderServer();
                
                // Set up a TCP listener for the order port
                // NinjaTrader now acts as server, listening for Python client
                Print($"Starting order server on {listenIP}:{orderPort}...");
                
                // Create server on loopback address
                IPAddress ipAddress = IPAddress.Parse(listenIP);
                orderServer = new TcpListener(ipAddress, orderPort);
                
                // Start listening
                orderServer.Start();
                Print($"Order server started on {listenIP}:{orderPort}");
                
                // Start the server thread to accept connections
                isOrderServerRunning = true;
                orderServerThread = new Thread(OrderServerLoop);
                orderServerThread.IsBackground = true;
                orderServerThread.Start();
                
                return true;
            }
            catch (Exception ex)
            {
                Print($"Error starting order server: {ex.Message}");
                
                // Cleanup in case of failure
                if (orderServer != null)
                {
                    try { orderServer.Stop(); } catch { }
                    orderServer = null;
                }
                
                isOrderServerRunning = false;
                return false;
            }
        }
        
        private void OrderServerLoop()
        {
            Print("Order server thread started. Waiting for Python client connection...");
            
            while (isOrderServerRunning)
            {
                try
                {
                    // Check for pending connections
                    if (orderServer.Pending())
                    {
                        // Accept the client connection
                        orderClient = orderServer.AcceptTcpClient();
                        orderClient.NoDelay = true;
                        orderClient.SendBufferSize = 65536;
                        orderClient.ReceiveBufferSize = 65536;
                        
                        // Get the client stream
                        orderClientStream = orderClient.GetStream();
                        
                        Print($"Python client connected to order server from {orderClient.Client.RemoteEndPoint}");
                        
                        // Send welcome message
                        SendOrderMessage("ORDER_SERVER_READY\n");
                        
                        // Process data from the client
                        byte[] buffer = new byte[4096];
                        StringBuilder messageBuilder = new StringBuilder();
                        
                        // Main client data reception loop
                        while (isOrderServerRunning && orderClient != null && orderClient.Connected)
                        {
                            // Check if there's data to read
                            if (orderClient.Available > 0)
                            {
                                int bytesRead = orderClientStream.Read(buffer, 0, buffer.Length);
                                
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
                                        ProcessOrderMessage(completeMessage);
                                    }
                                }
                                
                                // Update buffer with remaining incomplete message
                                messageBuilder.Clear();
                                messageBuilder.Append(message);
                            }
                            
                            // Short sleep to prevent high CPU usage
                            Thread.Sleep(10);
                        }
                        
                        // Client disconnected or error occurred
                        Print("Python client disconnected from order server");
                        
                        // Clean up client resources
                        if (orderClientStream != null)
                        {
                            try { orderClientStream.Close(); } catch { }
                            orderClientStream = null;
                        }
                        
                        if (orderClient != null)
                        {
                            try { orderClient.Close(); } catch { }
                            orderClient = null;
                        }
                    }
                    
                    // Short sleep before checking for new connections
                    Thread.Sleep(100);
                }
                catch (Exception ex)
                {
                    Print($"Error in order server loop: {ex.Message}");
                    
                    // Clean up client resources
                    if (orderClientStream != null)
                    {
                        try { orderClientStream.Close(); } catch { }
                        orderClientStream = null;
                    }
                    
                    if (orderClient != null)
                    {
                        try { orderClient.Close(); } catch { }
                        orderClient = null;
                    }
                    
                    // Short delay before retrying
                    Thread.Sleep(1000);
                }
            }
            
            Print("Order server thread stopped");
        }
        
        private void StopOrderServer()
        {
            try
            {
                // First mark thread as stopping
                isOrderServerRunning = false;
                
                // Close client streams and connections
                if (orderClientStream != null)
                {
                    try 
                    {
                        orderClientStream.Flush();
                        orderClientStream.Close();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error closing order client stream: {ex.Message}");
                    }
                    finally
                    {
                        orderClientStream = null;
                    }
                }
                
                if (orderClient != null)
                {
                    try
                    {
                        orderClient.Close();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error closing order client: {ex.Message}");
                    }
                    finally
                    {
                        orderClient = null;
                    }
                }
                
                // Stop the server
                if (orderServer != null)
                {
                    try
                    {
                        orderServer.Stop();
                    }
                    catch (Exception ex)
                    {
                        Print($"Error stopping order server: {ex.Message}");
                    }
                    finally
                    {
                        orderServer = null;
                    }
                }
                
                // Wait for thread to terminate
                if (orderServerThread != null && orderServerThread.IsAlive)
                {
                    try
                    {
                        orderServerThread.Join(2000);
                        if (orderServerThread.IsAlive)
                        {
                            // Not ideal but needed in case thread is stuck
                            orderServerThread.Abort();
                        }
                    }
                    catch { }
                    orderServerThread = null;
                }
                
                Print("Order server stopped successfully");
            }
            catch (Exception ex)
            {
                Print($"Error stopping order server: {ex.Message}");
            }
            finally
            {
                // Force resource cleanup
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }
        
        private void SendOrderMessage(string message)
        {
            try
            {
                // Check if connected to a client
                if (orderClient != null && orderClient.Connected && orderClientStream != null)
                {
                    // Send data directly
                    byte[] data = Encoding.ASCII.GetBytes(message);
                    orderClientStream.Write(data, 0, data.Length);
                }
            }
            catch (Exception ex)
            {
                Print("Error sending order message: " + ex.Message);
                
                // Client connection might be broken, clean up
                if (orderClientStream != null)
                {
                    try { orderClientStream.Close(); } catch { }
                    orderClientStream = null;
                }
                
                if (orderClient != null)
                {
                    try { orderClient.Close(); } catch { }
                    orderClient = null;
                }
            }
        }
        
        private void ProcessOrderMessage(string message)
        {
            try
            {
                if (string.IsNullOrEmpty(message))
                    return;
                
                Print("Received order message: " + message);
                
                // PING/HEARTBEAT handling to keep connection active
                if (message.Equals("PING", StringComparison.OrdinalIgnoreCase))
                {
                    // Respond with PONG to confirm connection 
                    try {
                        SendOrderMessage("PONG\n");
                    } catch (Exception ex) {
                        Print("Error sending PONG response: " + ex.Message);
                    }
                    return;
                }
                
                if (message.Equals("HEARTBEAT", StringComparison.OrdinalIgnoreCase))
                {
                    // Just to keep connection active, no response needed
                    return;
                }
                
                // Check for special command messages
                if (message.StartsWith(extractionCommand) || message.Equals("EXTRACT_HISTORICAL_DATA"))
                {
                    // Handle data extraction command
                    Print("Received data extraction command from Python. Starting extraction...");
                    this.Dispatcher.InvokeAsync(() => StartHistoricalDataExtraction());
                    return;
                }
                
                // Handle data extraction-related messages
                if (message.StartsWith("EXISTING_DATA_CHECK:") || message.Equals("EXTRACT_HISTORICAL_DATA"))
                {
                    string instrumentName = "Unknown";
                    if (message.StartsWith("EXISTING_DATA_CHECK:"))
                    {
                        string[] dataParts = message.Split(':');
                        if (dataParts.Length > 1)
                        {
                            instrumentName = dataParts[1];
                        }
                    }
                    
                    // Start historical data extraction 
                    this.Dispatcher.InvokeAsync(() => {
                        Print($"Python client requested historical data for {instrumentName}");
                        StartHistoricalDataExtraction();
                    });
                    
                    return;
                }
                
                // Process trading signals from Python
                string[] commandParts = message.Split(',');
                
                if (commandParts.Length < 5)
                {
                    Print("Invalid order message format. Expected at least 5 parameters.");
                    // Send error response to client
                    SendOrderMessage("ERROR:Invalid_Format\n");
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
                    SendOrderMessage("ERROR:Invalid_Values\n");
                    return;
                }
                
                // Validate parameters
                if (tradeSignal < -1 || tradeSignal > 1)
                {
                    Print("Invalid trade signal: " + tradeSignal);
                    SendOrderMessage("ERROR:Invalid_Signal\n");
                    return;
                }
                
                if (emaChoice < 0 || emaChoice > 2)
                {
                    Print("Invalid EMA choice: " + emaChoice);
                    SendOrderMessage("ERROR:Invalid_EMA\n");
                    return;
                }
                
                if (positionSize <= 0 || positionSize > 10)
                {
                    Print("Invalid position size: " + positionSize);
                    SendOrderMessage("ERROR:Invalid_Size\n");
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
                SendOrderMessage($"ORDER_CONFIRMED:{tradeSignal},{emaChoice},{positionSize}\n");
            }
            catch (Exception ex)
            {
                Print("Error processing order message: " + ex.Message);
                SendOrderMessage("ERROR:Processing_Error\n");
            }
        }
        
        // Public method to send trade execution data back to Python
        public void SendTradeExecutionUpdate(string action, double entryPrice, double exitPrice, double pnl, int quantity)
        {
            try
            {
                string executionMessage = $"TRADE_EXECUTED:{action},{entryPrice},{exitPrice},{pnl},{quantity}\n";
                SendOrderMessage(executionMessage);
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
            bool dataConnected = (dataClient != null && dataClient.Connected);
            bool orderConnected = (orderClient != null && orderClient.Connected);
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