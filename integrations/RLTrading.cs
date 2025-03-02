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
    public class RLTradeExecutor : Strategy
    {
        #region Variables
        private TcpClient dataSender;
        private NetworkStream dataSenderStream;
        private TcpListener orderReceiver;
        private NetworkStream orderReceiverStream;
        private string serverIP = "127.0.0.1";
        private int dataPort = 5000;
        private int orderPort = 5001;
        private DateTime lastConnectionAttempt = DateTime.MinValue;
        private bool isConnecting = false;
        private int reconnectInterval = 5; // seconds
        private int dataBufferSize = 60; // Number of bars to buffer
        private Queue<string> dataBuffer = new Queue<string>();
        private bool isInitialDataSent = false;
        private Thread orderReceiverThread;
        private bool isOrderReceiverRunning = false;
        private EMA emaShort;
        private EMA emaLong;
        private ATR atr;
        private ADX adx;
        private int lastMinute = -1;
        private bool waitForFullBar = true;
        
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
        private bool connectionErrorLogged = false; // Prevent log spam
        
        // Data extraction variables
        private bool isExtractingData = false;
        private bool isExtractionComplete = false;
        private int totalBarsToExtract = 0;
        private int extractedBarsCount = 0;
        private string extractionCommand = "EXTRACT_HISTORICAL_DATA";
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
        [Display(Name = "Base Position Size", Description = "Base number of contracts/units for trades", Order = 4, GroupName = "Trading Settings")]
        public double BasePositionSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Retry Count", Description = "Number of times to retry failed orders", Order = 5, GroupName = "Trading Settings")]
        public int OrderRetryCount { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Wait For Full Bar", Description = "Wait for full 1-minute bar to form before sending", Order = 6, GroupName = "Data Settings")]
        public bool WaitForFullBar { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Short Period", Description = "Period for short EMA calculation", Order = 7, GroupName = "Indicators")]
        public int EMAShortPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Long Period", Description = "Period for long EMA calculation", Order = 8, GroupName = "Indicators")]
        public int EMALongPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ATR Period", Description = "Period for ATR calculation", Order = 9, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ADX Period", Description = "Period for ADX calculation", Order = 10, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Historical Bars", Description = "Maximum number of historical bars to extract", Order = 11, GroupName = "Data Settings")]
        public int MaxHistoricalBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Connection Timeout", Description = "Timeout in seconds for socket connections", Order = 12, GroupName = "Socket Settings")]
        public int ConnectionTimeout { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Use Managed Orders", Description = "Use managed or unmanaged order handling", Order = 13, GroupName = "Trading Settings")]
        public bool UseManagedOrders { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Bidirectional communication with RL Python model for automated trading";
                Name = "RLTradeExecutor";
                
                // Default settings
                ServerIP = "127.0.0.1";
                DataPort = 5000;
                OrderPort = 5001;
                BasePositionSize = 1;
                OrderRetryCount = 3;
                WaitForFullBar = true;
                EMAShortPeriod = 9;
                EMALongPeriod = 21;
                ATRPeriod = 14;
                ADXPeriod = 14;
                MaxHistoricalBars = 6000; // Default maximum bars to extract
                ConnectionTimeout = 5;
                UseManagedOrders = true; // Default to managed orders
                
                // Set up calculation mode
                Calculate = Calculate.OnBarClose;
                BarsRequiredToTrade = 100;
                IsExitOnSessionCloseStrategy = true;
                
                // Set this according to UseManagedOrders
                IsUnmanaged = !UseManagedOrders;
            }
            else if (State == State.Configure)
            {
                // Set local variables from properties
                serverIP = ServerIP;
                dataPort = DataPort;
                orderPort = OrderPort;
                waitForFullBar = WaitForFullBar;
                basePositionSize = BasePositionSize;
                orderRetryCount = OrderRetryCount;
                
                // Update IsUnmanaged based on UseManagedOrders
                IsUnmanaged = !UseManagedOrders;
                
                // Add technical indicators
                emaShort = EMA(EMAShortPeriod);
                emaLong = EMA(EMALongPeriod);
                atr = ATR(ATRPeriod);
                adx = ADX(ADXPeriod);
                
                // Add indicators for calculations
                AddChartIndicator(emaShort);
                AddChartIndicator(emaLong);
                AddChartIndicator(atr);
                AddChartIndicator(adx);
            }
            else if (State == State.Realtime)
            {
                // Reset connection error flag
                connectionErrorLogged = false;
                
                // Initialize connections when switching to real-time
                InitializeDataSender();
                InitializeOrderReceiver();
            }
            else if (State == State.Terminated)
            {
                // Close connections when terminated
                CloseDataSender();
                CloseOrderReceiver();
            }
        }

        protected override void OnBarUpdate()
        {
            // Only process 1-minute bars on the primary series
            if (BarsInProgress != 0)
                return;
            
            // Check if we have enough bars
            if (CurrentBar < BarsRequiredToTrade)
                return;
            
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
            
            // If we are extracting historical data, check if we need to send more bars
            if (isExtractingData && !isExtractionComplete)
            {
                SendHistoricalBatch();
            }
        }

        #region Data Sender Methods
        private void InitializeDataSender()
        {
            if (isConnecting || (DateTime.Now - lastConnectionAttempt).TotalSeconds < reconnectInterval)
                return;
                
            isConnecting = true;
            lastConnectionAttempt = DateTime.Now;
            
            try
            {
                // Verificar si ya tenemos una conexión activa
                if (dataSender != null && dataSender.Connected && dataSenderStream != null && dataSenderStream.CanWrite)
                {
                    isConnecting = false;
                    
                    // Enviar PONG para verificar la conexión
                    try
                    {
                        byte[] pingData = Encoding.ASCII.GetBytes("PONG\n");
                        dataSenderStream.Write(pingData, 0, pingData.Length);
                    }
                    catch (Exception ex)
                    {
                        Print($"Error checking connection: {ex.Message}");
                        // Conexión está caída, cerrarla
                        CloseDataSender();
                    }
                    
                    return;
                }
                
                // Cerrar cualquier conexión existente
                CloseDataSender();
                
                // Crear un TcpClient primero - NinjaTrader actúa como cliente, conectándose a Python
                // Python estará actuando como servidor en este puerto
                Print($"Intentando conectar a Python en {serverIP}:{dataPort}...");
                
                // Esperar 3 segundos para dar tiempo a Python a iniciar su servidor
                Thread.Sleep(3000);
                
                // Crear nueva conexión
                dataSender = new TcpClient();
                
                // Configurar socket antes de la conexión
                dataSender.NoDelay = true;
                dataSender.SendBufferSize = 65536;
                dataSender.ReceiveBufferSize = 65536;
                dataSender.ReceiveTimeout = 5000;
                dataSender.SendTimeout = 5000;
                
                // Intentar conexión con timeout
                IAsyncResult result = dataSender.BeginConnect(serverIP, dataPort, null, null);
                bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(ConnectionTimeout));
                
                if (!success)
                {
                    dataSender.Close();
                    dataSender = null;
                    
                    // Registrar el error de conexión
                    if (!connectionErrorLogged)
                    {
                        Print("Error connecting to Python server: Connection timeout");
                        connectionErrorLogged = true;
                    }
                    
                    isConnecting = false;
                    return;
                }
                
                // Completar la conexión
                dataSender.EndConnect(result);
                
                // Breve espera para asegurar que la conexión esté lista
                Thread.Sleep(200);
                
                // Obtener el stream
                dataSenderStream = dataSender.GetStream();
                
                // Enviar handshake inicial
                byte[] handshake = Encoding.ASCII.GetBytes("PONG\n");
                
                try
                {
                    dataSenderStream.Write(handshake, 0, handshake.Length);
                    Print("Handshake enviado a Python");
                }
                catch (Exception ex)
                {
                    Print($"Error al enviar handshake: {ex.Message}");
                    CloseDataSender();
                    isConnecting = false;
                    return;
                }
                
                // Conexión exitosa, resetear flag de error
                connectionErrorLogged = false;
                Print("Connected to Python server at " + serverIP + ":" + dataPort);
                
                // Enviar datos acumulados
                while (dataBuffer.Count > 0)
                {
                    try
                    {
                        string data = dataBuffer.Dequeue();
                        byte[] buffer = Encoding.ASCII.GetBytes(data);
                        dataSenderStream.Write(buffer, 0, buffer.Length);
                    }
                    catch (Exception ex)
                    {
                        Print($"Error sending buffered data: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                if (!connectionErrorLogged)
                {
                    Print("Error connecting to Python server: " + ex.Message);
                    connectionErrorLogged = true;
                }
                
                // Asegurar limpieza si falló la conexión
                if (dataSenderStream != null)
                {
                    try { dataSenderStream.Close(); } catch { }
                    dataSenderStream = null;
                }
                
                if (dataSender != null)
                {
                    try { dataSender.Close(); } catch { }
                    dataSender = null;
                }
            }
            finally
            {
                isConnecting = false;
            }
        }

        private void CloseDataSender()
        {
            try
            {
                // Cerrar primero el stream
                if (dataSenderStream != null)
                {
                    try 
                    {
                        dataSenderStream.Flush();
                        dataSenderStream.Close();
                    }
                    catch (Exception streamEx)
                    {
                        Print("Error closing data stream: " + streamEx.Message);
                    }
                    finally
                    {
                        dataSenderStream = null;
                    }
                }
                
                // Luego cerrar el cliente TCP
                if (dataSender != null)
                {
                    try
                    {
                        dataSender.Close();
                    }
                    catch (Exception tcpEx)
                    {
                        Print("Error closing TCP client: " + tcpEx.Message);
                    }
                    finally
                    {
                        dataSender = null;
                    }
                }
                
                Print("Data sender closed successfully");
            }
            catch (Exception ex)
            {
                Print("Error closing data sender connection: " + ex.Message);
            }
            finally
            {
                // Forzar limpieza de recursos
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // Method for extracting historical data on demand
        private void StartHistoricalDataExtraction()
        {
            if (isExtractingData)
            {
                Print("Historical data extraction already in progress");
                return;
            }
            
            int availableBars = Math.Min(CurrentBar, MaxHistoricalBars);
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
            
            Print($"Starting extraction of {totalBarsToExtract} historical bars for {instrumentName}...");
            
            // Send a message to check existing data first
            string checkMessage = $"CHECK_EXISTING_DATA:{instrumentName}\n";
            SendDataMessage(checkMessage);
            
            // Wait for response from Python about existing data
            // The actual data sending will be triggered when Python responds
            Print($"Checking for existing data for {instrumentName}...");
        }
        
        // Variable para almacenar la fecha desde la que filtrar los datos
        private DateTime? startFromDate = null;
        
        private void SendHistoricalBatch()
        {
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
            
            for (int i = 0; i < barsToSend; i++)
            {
                int barIndex = totalBarsToExtract - extractedBarsCount - sentInThisBatch - 1;
                if (barIndex >= 0 && barIndex <= CurrentBar)
                {
                    // Check if we need to filter by date
                    if (startFromDate.HasValue)
                    {
                        // If bar time is AFTER our start date, send the data
                        if (Time[barIndex] > startFromDate.Value)
                        {
                            SendBarDataForExtraction(barIndex);
                            sentInThisBatch++;
                        }
                    }
                    else
                    {
                        // No date filtering, send all data
                        SendBarDataForExtraction(barIndex);
                        sentInThisBatch++;
                    }
                }
            }
            
            // Update the count of extracted bars
            extractedBarsCount += sentInThisBatch;
            
            // Send progress update
            double progressPercent = (double)extractedBarsCount / totalBarsToExtract * 100;
            string progressMessage = $"EXTRACTION_PROGRESS:{extractedBarsCount}:{totalBarsToExtract}:{progressPercent:F1}\n";
            SendDataMessage(progressMessage);
            
            Print($"Extraction progress: {progressPercent:F1}% ({extractedBarsCount}/{totalBarsToExtract})");
        }
        
        private void SendBarDataForExtraction(int barIndex)
        {
            if (barIndex > CurrentBar)
                return;
                
            try
            {
                // Prepare data message with OHLC and indicators
                string message = string.Format("HISTORICAL:{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                    Open[barIndex].ToString("F2"),
                    High[barIndex].ToString("F2"),
                    Low[barIndex].ToString("F2"),
                    Close[barIndex].ToString("F2"),
                    emaShort[barIndex].ToString("F2"),
                    emaLong[barIndex].ToString("F2"),
                    atr[barIndex].ToString("F4"),
                    adx[barIndex].ToString("F2"),
                    Time[barIndex].ToString("yyyy-MM-dd HH:mm:ss"));
                
                // Add newline for message separation
                message += "\n";
                
                // Send the message
                SendDataMessage(message);
            }
            catch (Exception ex)
            {
                Print($"Error sending historical bar data: {ex.Message}");
            }
        }

        private void SendBarData(int barIndex)
        {
            if (barIndex > CurrentBar)
                return;
                
            try
            {
                // Prepare data message with OHLC and indicators
                string message = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8}",
                    Open[barIndex].ToString("F2"),
                    High[barIndex].ToString("F2"),
                    Low[barIndex].ToString("F2"),
                    Close[barIndex].ToString("F2"),
                    emaShort[barIndex].ToString("F2"),
                    emaLong[barIndex].ToString("F2"),
                    atr[barIndex].ToString("F4"),
                    adx[barIndex].ToString("F2"),
                    Time[barIndex].ToString("yyyy-MM-dd HH:mm:ss"));
                
                // Add newline for message separation
                message += "\n";
                
                // Send the message
                SendDataMessage(message);
            }
            catch (Exception ex)
            {
                Print("Error sending data: " + ex.Message);
            }
        }
        
        private void SendDataMessage(string message)
        {
            try
            {
                // Check if connected
                if (dataSender != null && dataSender.Connected && dataSenderStream != null && dataSenderStream.CanWrite)
                {
                    // Send data directly
                    byte[] data = Encoding.ASCII.GetBytes(message);
                    dataSenderStream.Write(data, 0, data.Length);
                }
                else
                {
                    // Buffer the data
                    dataBuffer.Enqueue(message);
                    
                    // Limit buffer size
                    while (dataBuffer.Count > dataBufferSize)
                    {
                        dataBuffer.Dequeue();
                    }
                    
                    // Try to reconnect
                    InitializeDataSender();
                }
            }
            catch (Exception ex)
            {
                Print("Error sending data message: " + ex.Message);
                // Try to reconnect on next update
                CloseDataSender();
            }
        }
        #endregion

        #region Order Receiver Methods
        private void InitializeOrderReceiver()
        {
            try
            {
                if (orderReceiver != null)
                    return;

                // Initialize order receiver
                IPAddress ipAddress = IPAddress.Parse(serverIP);
                orderReceiver = new TcpListener(ipAddress, orderPort);
                orderReceiver.Start();
                
                Print("Order receiver started on " + serverIP + ":" + orderPort);
                
                // Start the order receiver thread
                isOrderReceiverRunning = true;
                orderReceiverThread = new Thread(ReceiveOrders);
                orderReceiverThread.IsBackground = true;
                orderReceiverThread.Start();
            }
            catch (Exception ex)
            {
                Print("Error initializing order receiver: " + ex.Message);
            }
        }

        private void CloseOrderReceiver()
        {
            try
            {
                // Set the flag to stop the thread
                isOrderReceiverRunning = false;
                
                // Properly terminate the thread
                if (orderReceiverThread != null && orderReceiverThread.IsAlive)
                {
                    try
                    {
                        // Give it some time to finish gracefully
                        orderReceiverThread.Join(2000); // Wait up to 2 seconds for thread to finish
                        
                        if (orderReceiverThread.IsAlive)
                        {
                            // Only abort if necessary - this is a last resort
                            // as Thread.Abort() can lead to resource leaks
                            orderReceiverThread.Abort();
                            
                            // Wait a bit for the abort to take effect
                            Thread.Sleep(300);
                        }
                    }
                    catch (Exception threadEx)
                    {
                        Print("Error shutting down order receiver thread: " + threadEx.Message);
                    }
                    
                    orderReceiverThread = null;
                }
                
                // Close the network stream
                if (orderReceiverStream != null)
                {
                    try
                    {
                        orderReceiverStream.Close();
                    }
                    catch { }
                    orderReceiverStream = null;
                }
                
                // Stop the TCP listener
                if (orderReceiver != null)
                {
                    try
                    {
                        orderReceiver.Stop();
                    }
                    catch { }
                    orderReceiver = null;
                }
                
                Print("Order receiver closed successfully");
            }
            catch (Exception ex)
            {
                Print("Error closing order receiver: " + ex.Message);
            }
            finally
            {
                // Force garbage collection to clean up resources
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        private void ReceiveOrders()
        {
            while (isOrderReceiverRunning)
            {
                try
                {
                    TcpClient client = null;
                    
                    // Use a timeout approach to allow for thread cancellation checks
                    if (orderReceiver.Pending())
                    {
                        client = orderReceiver.AcceptTcpClient();
                        Print("Order client connected");
                    }
                    else
                    {
                        Thread.Sleep(100);
                        continue;
                    }
                    
                    NetworkStream clientStream = client.GetStream();
                    
                    // Buffer for receiving data
                    byte[] buffer = new byte[4096];
                    StringBuilder messageBuilder = new StringBuilder();
                    
                    while (isOrderReceiverRunning && client.Connected)
                    {
                        // Check for available data with timeout
                        if (client.Available > 0)
                        {
                            int bytesRead = clientStream.Read(buffer, 0, buffer.Length);
                            
                            if (bytesRead == 0)
                            {
                                // Client disconnected
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
                                
                                // Remove from buffer
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
                        
                        // Small delay to prevent CPU hogging
                        Thread.Sleep(10);
                    }
                    
                    // Clean up client resources
                    if (clientStream != null)
                        clientStream.Close();
                    
                    if (client != null)
                        client.Close();
                    
                    Print("Order client disconnected");
                }
                catch (ThreadAbortException)
                {
                    // Thread being aborted, exit gracefully
                    break;
                }
                catch (Exception ex)
                {
                    Print("Error in order receiver: " + ex.Message);
                    Thread.Sleep(1000); // Delay before retry
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
                
                // PING/HEARTBEAT handling para mantener la conexión activa
                if (message.Equals("PING", StringComparison.OrdinalIgnoreCase))
                {
                    // Responder con PONG para confirmar conexión 
                    try {
                        SendDataMessage("PONG\n");
                    } catch (Exception ex) {
                        Print("Error sending PONG response: " + ex.Message);
                    }
                    return;
                }
                
                if (message.Equals("HEARTBEAT", StringComparison.OrdinalIgnoreCase))
                {
                    // Solo para mantener la conexión activa, no es necesaria respuesta
                    return;
                }
                
                // Check for special command messages
                if (message.StartsWith(extractionCommand))
                {
                    // Handle data extraction command
                    this.Dispatcher.InvokeAsync(() => StartHistoricalDataExtraction());
                    return;
                }
                
                // Handle responses from Python server about existing data
                if (message.StartsWith("EXISTING_DATA_FOUND:"))
                {
                    string[] dataParts = message.Split(':');
                    if (dataParts.Length > 2)
                    {
                        string instrumentName = dataParts[1];
                        string dateTimeStr = dataParts[2];
                        
                        Print($"Found existing data for {instrumentName} up to {dateTimeStr}");
                        
                        // Parse the date string to DateTime
                        DateTime startDate;
                        if (DateTime.TryParse(dateTimeStr, out startDate))
                        {
                            // Set the start date for filtering
                            startFromDate = startDate;
                            Print($"Will extract data after {startDate}");
                        }
                        else
                        {
                            Print($"Could not parse date: {dateTimeStr}, will extract all data");
                            startFromDate = null;
                        }
                        
                        // Send start message with total bars and the date to start from
                        string startMessage = $"EXTRACTION_START:{totalBarsToExtract}:{instrumentName}:{dateTimeStr}\n";
                        SendDataMessage(startMessage);
                        
                        // Start sending historical data (filtered by date)
                        this.Dispatcher.InvokeAsync(() => SendHistoricalBatch());
                    }
                    return;
                }
                else if (message.StartsWith("NO_EXISTING_DATA:"))
                {
                    string[] dataParts = message.Split(':');
                    if (dataParts.Length > 1)
                    {
                        string instrumentName = dataParts[1];
                        Print($"No existing data found for {instrumentName}. Starting fresh extraction...");
                        
                        // Reset start date for a fresh extraction
                        startFromDate = null;
                        
                        // Start fresh extraction
                        string startMessage = $"EXTRACTION_START:{totalBarsToExtract}:{instrumentName}\n";
                        SendDataMessage(startMessage);
                        
                        // Start sending historical data
                        this.Dispatcher.InvokeAsync(() => SendHistoricalBatch());
                    }
                    return;
                }
                
                string[] commandParts = message.Split(',');
                
                if (commandParts.Length < 5)
                {
                    Print("Invalid order message format. Expected at least 5 parameters.");
                    // Enviar respuesta de error al cliente
                    SendDataMessage("ERROR:Invalid_Format\n");
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
                    SendDataMessage("ERROR:Invalid_Values\n");
                    return;
                }
                
                // Validate parameters
                if (tradeSignal < -1 || tradeSignal > 1)
                {
                    Print("Invalid trade signal: " + tradeSignal);
                    SendDataMessage("ERROR:Invalid_Signal\n");
                    return;
                }
                
                if (emaChoice < 0 || emaChoice > 2)
                {
                    Print("Invalid EMA choice: " + emaChoice);
                    SendDataMessage("ERROR:Invalid_EMA\n");
                    return;
                }
                
                if (positionSize <= 0 || positionSize > 10)
                {
                    Print("Invalid position size: " + positionSize);
                    SendDataMessage("ERROR:Invalid_Size\n");
                    return;
                }
                
                // Store current trading parameters
                currentSignal = tradeSignal;
                currentEMA = emaChoice;
                currentPositionSize = positionSize;
                currentStopLoss = stopLoss;
                currentTakeProfit = takeProfit;
                
                // Execute the trading decision on the main thread
                this.Dispatcher.InvokeAsync(() => ExecuteTradingDecision());
                
                // Confirmar que se procesó correctamente el mensaje
                SendDataMessage($"ORDER_CONFIRMED:{tradeSignal},{emaChoice},{positionSize}\n");
            }
            catch (Exception ex)
            {
                Print("Error processing order message: " + ex.Message);
                SendDataMessage("ERROR:Processing_Error\n");
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
        #endregion

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            try
            {
                // Log execution details
                Print($"Order executed: {marketPosition} {quantity} @ {price:F2}, OrderId: {orderId}");
                
                // Additional logic to track trade performance could be added here
                // For example, calculating P&L, updating statistics, etc.
                
                // Example: Send trade notification back to Python server
                // This would require a separate connection/socket
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
                else if (orderState == OrderState.Filled || orderState == OrderState.PartFilled)
                {
                    Print($"Order {order.Id} {orderState}: Filled {filled} @ {averageFillPrice:F2}");
                    
                    // Determine action and P&L for sending to Python
                    string action = "";
                    double entryPrice = 0;
                    double exitPrice = 0;
                    double pnl = 0;
                    
                    // Determine trade type from order name
                    if (order.Name.Contains("RL_Long") && !order.Name.Contains("Exit") && !order.Name.Contains("SL") && !order.Name.Contains("TP"))
                    {
                        action = "Enter Long";
                        entryPrice = averageFillPrice;
                    }
                    else if (order.Name.Contains("RL_Short") && !order.Name.Contains("Exit") && !order.Name.Contains("SL") && !order.Name.Contains("TP"))
                    {
                        action = "Enter Short";
                        entryPrice = averageFillPrice;
                    }
                    else if (order.Name.Contains("Exit") || order.Name.Contains("SL") || order.Name.Contains("TP"))
                    {
                        // This is an exit order
                        if (order.Name.Contains("Long"))
                        {
                            action = "Exit Long";
                            exitPrice = averageFillPrice;
                            // Calculate P&L for long position
                            double positionEntryPrice = Position.AveragePrice;
                            pnl = (exitPrice - positionEntryPrice) * filled * Instrument.MasterInstrument.PointValue;
                        }
                        else if (order.Name.Contains("Short"))
                        {
                            action = "Exit Short";
                            exitPrice = averageFillPrice;
                            // Calculate P&L for short position
                            double positionEntryPrice = Position.AveragePrice;
                            pnl = (positionEntryPrice - exitPrice) * filled * Instrument.MasterInstrument.PointValue;
                        }
                    }
                    
                    // Send execution data to Python
                    if (!string.IsNullOrEmpty(action))
                    {
                        string executionMessage = $"TRADE_EXECUTED:{action},{entryPrice},{exitPrice},{pnl},{filled}\n";
                        SendDataMessage(executionMessage);
                    }
                }
                else if (orderState == OrderState.Cancelled)
                {
                    Print($"Order {order.Id} cancelled");
                }
            }
            catch (Exception ex)
            {
                Print($"Error in OnOrderUpdate: {ex.Message}");
            }
        }
    }
}