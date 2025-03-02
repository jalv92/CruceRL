using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Controls.Primitives;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Xml.Serialization;

namespace NinjaTrader.NinjaScript.Indicators
{
    /// <summary>
    /// This indicator extracts market data and indicator values for a Python-based RL trading system.
    /// Features a control panel with an "Extract Data" button for manual data extraction.
    /// </summary>
    public class RLLink : Indicator
    {
        #region Variables
        // Data extraction variables
        private bool isExtractingData = false;
        private bool isExtractionComplete = false;
        private int totalBarsToExtract = 0;
        private int extractedBarsCount = 0;
        
        // CSV export variables
        private string csvFilePath = "";
        private StreamWriter csvWriter = null;
        
        // Technical indicators
        private EMA emaShort;
        private EMA emaLong;
        private ATR atr;
        private ADX adx;
        private RSI rsi;
        private Bollinger bollinger;
        private MACD macd;
        
        // UI Elements
        private Grid controlPanel;
        private Button extractButton;
        private TextBlock statusTextBlock;
        private System.Windows.Shapes.Rectangle progressBar;
        private System.Windows.Shapes.Rectangle progressBarBackground;
        private bool controlPanelVisible = true;
        private string statusMessage = "Ready";
        private double progressPercent = 0;
        
        // Panel settings
        private int panelWidth = 175;
        private int panelHeight = 95;
        private int panelMargin = 10;
        private SolidColorBrush panelBackground = new SolidColorBrush(Color.FromArgb(200, 30, 30, 50));
        private SolidColorBrush buttonColor = new SolidColorBrush(Color.FromArgb(255, 55, 120, 180));
        private SolidColorBrush buttonHoverColor = new SolidColorBrush(Color.FromArgb(255, 70, 150, 220));
        private SolidColorBrush buttonTextColor = new SolidColorBrush(Colors.White);
        private SolidColorBrush statusTextColor = new SolidColorBrush(Colors.LightGray);
        private SolidColorBrush progressBarColor = new SolidColorBrush(Color.FromArgb(255, 46, 204, 113));
        private SolidColorBrush progressBarBackgroundColor = new SolidColorBrush(Color.FromArgb(80, 150, 150, 150));
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Export Folder Path", Description = "Folder path for CSV export (use full path)", Order = 1, GroupName = "Export Settings")]
        public string ExportFolderPath { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "EMA Short Period", Description = "Period for short EMA calculation", Order = 2, GroupName = "Indicators")]
        public int EMAShortPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Long Period", Description = "Period for long EMA calculation", Order = 3, GroupName = "Indicators")]
        public int EMALongPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ATR Period", Description = "Period for ATR calculation", Order = 4, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ADX Period", Description = "Period for ADX calculation", Order = 5, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "RSI Period", Description = "Period for RSI calculation", Order = 6, GroupName = "Indicators")]
        public int RSIPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Bollinger Bands Period", Description = "Period for Bollinger Bands calculation", Order = 7, GroupName = "Indicators")]
        public int BollingerPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Bollinger Bands StdDev", Description = "Standard deviations for Bollinger Bands", Order = 8, GroupName = "Indicators")]
        public double BollingerStdDev { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "MACD Fast", Description = "MACD fast period", Order = 9, GroupName = "Indicators")]
        public int MACDFast { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "MACD Slow", Description = "MACD slow period", Order = 10, GroupName = "Indicators")]
        public int MACDSlow { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "MACD Signal", Description = "MACD signal period", Order = 11, GroupName = "Indicators")]
        public int MACDSignal { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Historical Bars", Description = "Maximum number of historical bars to extract", Order = 12, GroupName = "Data Settings")]
        public int MaxHistoricalBars { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Show Control Panel", Description = "Show or hide the control panel", Order = 13, GroupName = "Interface")]
        public bool ShowControlPanel { get; set; }
        
        [Browsable(false)]
        [XmlIgnore]
        public Series<bool> IsExtracting { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Data extraction indicator for RL Python model with control panel";
                Name = "RLLink";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;  // Set to true so we can display the control panel
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                PaintPriceMarkers = true;
                
                // Default settings
                ExportFolderPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\NT8 RL\\data";
                EMAShortPeriod = 9;
                EMALongPeriod = 21;
                ATRPeriod = 14;
                ADXPeriod = 14;
                RSIPeriod = 14;
                BollingerPeriod = 20;
                BollingerStdDev = 2;
                MACDFast = 12;
                MACDSlow = 26;
                MACDSignal = 9;
                MaxHistoricalBars = 2000;
                ShowControlPanel = true;
            }
            else if (State == State.Configure)
            {
                // Add the series we want to plot
                IsExtracting = new Series<bool>(this);
                
                // Add technical indicators
                emaShort = EMA(EMAShortPeriod);
                emaLong = EMA(EMALongPeriod);
                atr = ATR(ATRPeriod);
                adx = ADX(ADXPeriod);
                rsi = RSI(RSIPeriod, 3);
                bollinger = Bollinger(BollingerPeriod, (int)BollingerStdDev);
                macd = MACD(MACDFast, MACDSlow, MACDSignal);
                
                // Add extraction status indicator
                AddPlot(new Stroke(System.Windows.Media.Brushes.DodgerBlue, 1), PlotStyle.Line, "ExtractionStatus");
            }
            else if (State == State.DataLoaded)
            {
                // Create the control panel when data is loaded
                if (ShowControlPanel && ChartControl != null)
                {
                    CreateControlPanel();
                }
            }
            else if (State == State.Terminated)
            {
                // Clean up resources and UI elements
                RemoveControlPanel();
                CloseCSVFile();
            }
        }
        
        protected override void OnBarUpdate()
        {
            try
            {
                // Only process bars on the primary series
                if (BarsInProgress != 0)
                    return;
                
                // Check if we have enough bars
                if (CurrentBar < 20) // Need at least 20 bars for indicators to work
                    return;
                
                // Update the extraction status plot
                IsExtracting[0] = isExtractingData;
                Values[0][0] = isExtractingData ? 1 : 0;
                
                // If we are extracting historical data, process that first
                if (isExtractingData && !isExtractionComplete)
                {
                    // Process extraction in a background thread to avoid UI freezing
                    this.Dispatcher.InvokeAsync(() => {
                        try {
                            ProcessHistoricalBatch();
                        }
                        catch (Exception ex) {
                            Print($"Error in ProcessHistoricalBatch: {ex.Message}");
                            CompleteExtraction(); // Ensure we clean up if there's an error
                        }
                    });
                }
                
                // Update progress bar if extraction is in progress
                if (isExtractingData && progressBar != null)
                {
                    this.Dispatcher.InvokeAsync(() => {
                        UpdateProgressBar(progressPercent);
                    });
                }
            }
            catch (Exception ex)
            {
                Print($"Error in OnBarUpdate: {ex.Message}");
            }
        }
        
        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);
            
            // Update control panel position if it's visible
            if (controlPanel != null && controlPanel.Visibility == Visibility.Visible)
            {
                // Position in top right corner
                Canvas.SetLeft(controlPanel, chartControl.ActualWidth - panelWidth - panelMargin);
                Canvas.SetTop(controlPanel, panelMargin);
            }
        }
        
        #region UI Methods
        /// <summary>
        /// Creates the control panel UI
        /// </summary>
        private void CreateControlPanel()
        {
            try
            {
                if (ChartControl == null)
                    return;
                    
                // Remove existing panel if any
                RemoveControlPanel();
                
                // Create main panel
                controlPanel = new Grid();
                controlPanel.Width = panelWidth;
                controlPanel.Height = panelHeight;
                controlPanel.Background = panelBackground;
                controlPanel.Visibility = ShowControlPanel ? Visibility.Visible : Visibility.Collapsed;
                
                // Create panel header
                Border headerBorder = new Border();
                headerBorder.Height = 24;
                headerBorder.Background = new SolidColorBrush(Color.FromArgb(255, 40, 40, 60));
                headerBorder.VerticalAlignment = VerticalAlignment.Top;
                
                TextBlock headerText = new TextBlock();
                headerText.Text = "RL Control Panel";
                headerText.Foreground = buttonTextColor;
                headerText.FontWeight = FontWeights.Bold;
                headerText.VerticalAlignment = VerticalAlignment.Center;
                headerText.HorizontalAlignment = HorizontalAlignment.Center;
                
                headerBorder.Child = headerText;
                controlPanel.Children.Add(headerBorder);
                
                // Define rows for layout
                controlPanel.RowDefinitions.Add(new RowDefinition { Height = new GridLength(24) }); // Header
                controlPanel.RowDefinitions.Add(new RowDefinition { Height = new GridLength(40) }); // Button
                controlPanel.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) }); // Status + Progress
                
                // Create extract button
                extractButton = new Button();
                extractButton.Content = "Extract Data";
                extractButton.Background = buttonColor;
                extractButton.Foreground = buttonTextColor;
                extractButton.BorderThickness = new Thickness(0);
                extractButton.Margin = new Thickness(10, 8, 10, 8);
                extractButton.Click += ExtractButton_Click;
                extractButton.MouseEnter += (s, e) => { extractButton.Background = buttonHoverColor; };
                extractButton.MouseLeave += (s, e) => { extractButton.Background = buttonColor; };
                
                Grid.SetRow(extractButton, 1);
                controlPanel.Children.Add(extractButton);
                
                // Create status text and progress bar container
                Grid statusGrid = new Grid();
                statusGrid.Margin = new Thickness(10, 0, 10, 8);
                Grid.SetRow(statusGrid, 2);
                
                // Status text
                statusTextBlock = new TextBlock();
                statusTextBlock.Text = statusMessage;
                statusTextBlock.Foreground = statusTextColor;
                statusTextBlock.VerticalAlignment = VerticalAlignment.Top;
                statusTextBlock.Margin = new Thickness(0, 0, 0, 3);
                statusGrid.Children.Add(statusTextBlock);
                
                // Progress bar background
                progressBarBackground = new System.Windows.Shapes.Rectangle();
                progressBarBackground.Height = 6;
                progressBarBackground.Fill = progressBarBackgroundColor;
                progressBarBackground.RadiusX = 3;
                progressBarBackground.RadiusY = 3;
                progressBarBackground.VerticalAlignment = VerticalAlignment.Bottom;
                statusGrid.Children.Add(progressBarBackground);
                
                // Progress bar
                progressBar = new System.Windows.Shapes.Rectangle();
                progressBar.Height = 6;
                progressBar.Fill = progressBarColor;
                progressBar.RadiusX = 3;
                progressBar.RadiusY = 3;
                progressBar.VerticalAlignment = VerticalAlignment.Bottom;
                progressBar.HorizontalAlignment = HorizontalAlignment.Left;
                progressBar.Width = 0; // Start with no progress
                statusGrid.Children.Add(progressBar);
                
                controlPanel.Children.Add(statusGrid);
                
                // Add panel to chart
                ChartControl.Dispatcher.InvokeAsync(new Action(() => {
                    try {
                        ChartControl.Parent.Controls.Add(controlPanel);
                        Canvas.SetLeft(controlPanel, ChartControl.ActualWidth - panelWidth - panelMargin);
                        Canvas.SetTop(controlPanel, panelMargin);
                        Canvas.SetZIndex(controlPanel, 1000); // Ensure it's on top
                    }
                    catch (Exception ex) {
                        Print($"Error adding control panel to chart: {ex.Message}");
                    }
                }));
            }
            catch (Exception ex)
            {
                Print($"Error creating control panel: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Removes the control panel from the chart
        /// </summary>
        private void RemoveControlPanel()
        {
            if (ChartControl != null && controlPanel != null)
            {
                ChartControl.Dispatcher.InvokeAsync(() => {
                    try {
                        if (ChartControl.Parent != null && ChartControl.Parent.Controls.Contains(controlPanel))
                        {
                            ChartControl.Parent.Controls.Remove(controlPanel);
                        }
                        
                        // Clean up event handlers
                        if (extractButton != null)
                        {
                            extractButton.Click -= ExtractButton_Click;
                        }
                        
                        controlPanel = null;
                        extractButton = null;
                        statusTextBlock = null;
                        progressBar = null;
                        progressBarBackground = null;
                    }
                    catch (Exception ex) {
                        Print($"Error removing control panel: {ex.Message}");
                    }
                });
            }
        }
        
        /// <summary>
        /// Updates the status message and progress bar
        /// </summary>
        private void UpdateStatus(string message, double percent)
        {
            if (ChartControl == null || statusTextBlock == null)
                return;
                
            try
            {
                statusMessage = message;
                progressPercent = percent;
                
                ChartControl.Dispatcher.InvokeAsync(() => {
                    try {
                        if (statusTextBlock != null)
                            statusTextBlock.Text = message;
                            
                        UpdateProgressBar(percent);
                    }
                    catch { }
                });
            }
            catch { }
        }
        
        /// <summary>
        /// Updates just the progress bar
        /// </summary>
        private void UpdateProgressBar(double percent)
        {
            if (progressBar != null && progressBarBackground != null)
            {
                try
                {
                    // Clamp to 0-100%
                    percent = Math.Max(0, Math.Min(100, percent));
                    double width = (progressBarBackground.Width * percent) / 100.0;
                    progressBar.Width = width;
                }
                catch { }
            }
        }
        
        /// <summary>
        /// Event handler for Extract button click
        /// </summary>
        private void ExtractButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (isExtractingData)
                {
                    // If extraction is already running, don't start another one
                    UpdateStatus("Extraction in progress...", progressPercent);
                    return;
                }
                
                // Update UI to show we're starting extraction
                UpdateStatus("Starting extraction...", 0);
                
                // Disable the button during extraction
                if (extractButton != null)
                    extractButton.IsEnabled = false;
                
                // Start the extraction process
                StartHistoricalDataExtraction();
            }
            catch (Exception ex)
            {
                Print($"Error handling extract button click: {ex.Message}");
                UpdateStatus("Extraction failed", 0);
                
                // Re-enable button
                if (extractButton != null)
                    extractButton.IsEnabled = true;
            }
        }
        #endregion
        
        #region Data Extraction Methods
        /// <summary>
        /// Starts extraction of historical data to CSV
        /// </summary>
        public void StartHistoricalDataExtraction(int requestedBars = 0)
        {
            try
            {
                // Don't start a new extraction if one is already running
                if (isExtractingData)
                {
                    Print("Historical data extraction already in progress");
                    return;
                }
                
                // Ensure we're in realtime mode for extraction
                if (State != State.Realtime && State != State.Historical)
                {
                    Print($"ERROR: Indicator must be in Realtime or Historical state for extraction. Current state: {State}");
                    UpdateStatus("Error: Incorrect state", 0);
                    return;
                }
                
                // Make sure any previous resources are cleaned up
                CompleteExtraction();
                
                // Check if indicators are ready
                bool hasValidIndicators = (emaShort != null && emaLong != null && atr != null && adx != null);
                
                if (!hasValidIndicators)
                {
                    Print("ERROR: Required indicators not initialized. Cannot extract data.");
                    UpdateStatus("Error: Indicators not ready", 0);
                    return;
                }
                
                // Validate that we have enough bars
                int barsAvailable = CurrentBar + 1;
                Print($"Total bars available in chart: {barsAvailable}");
                
                if (barsAvailable < 20)
                {
                    Print("ERROR: Not enough bars loaded in chart. Need at least 20 bars.");
                    UpdateStatus("Error: Not enough bars", 0);
                    return;
                }
                
                // Determine number of bars to extract
                int barsRequested = requestedBars > 0 ? requestedBars : MaxHistoricalBars;
                int barsToExtract = Math.Min(barsRequested, barsAvailable);
                
                // Limit to a reasonable number to prevent errors
                if (barsToExtract > 3000)
                {
                    Print($"WARNING: Limiting extraction to 3000 bars (requested {barsToExtract}) for performance.");
                    barsToExtract = 3000;
                }
                
                // Prepare CSV file
                string instrumentName = Instrument.MasterInstrument.Name;
                PrepareCSVFile(instrumentName);
                
                if (csvWriter == null)
                {
                    Print("ERROR: Failed to create CSV file. Cannot extract data.");
                    UpdateStatus("Error: Failed to create file", 0);
                    return;
                }
                
                // Initialize extraction state
                isExtractingData = true;
                isExtractionComplete = false;
                totalBarsToExtract = barsToExtract;
                extractedBarsCount = 0;
                
                Print($"Starting extraction of {totalBarsToExtract} historical bars for {instrumentName}...");
                UpdateStatus("Extracting data...", 0);
                
                // Process the first batch
                this.Dispatcher.InvokeAsync(() => {
                    try {
                        ProcessHistoricalBatch();
                    }
                    catch (Exception ex) {
                        Print($"ERROR: Exception starting extraction: {ex.Message}");
                        CompleteExtraction();
                    }
                });
            }
            catch (Exception ex)
            {
                Print($"ERROR: Failed to start historical data extraction: {ex.Message}");
                isExtractingData = false;
                isExtractionComplete = true;
                UpdateStatus("Extraction failed", 0);
                
                // Re-enable the button
                if (extractButton != null)
                    extractButton.IsEnabled = true;
            }
        }
        
        /// <summary>
        /// Prepares the CSV file for extraction
        /// </summary>
        private void PrepareCSVFile(string instrumentName)
        {
            try
            {
                // Make sure the export directory exists
                string exportDir = ExportFolderPath;
                if (!Directory.Exists(exportDir))
                {
                    try
                    {
                        Directory.CreateDirectory(exportDir);
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR: Failed to create export directory: {ex.Message}");
                        return;
                    }
                }
                
                // Ensure any previous writer is closed
                CloseCSVFile();
                
                // Create the CSV file path with timestamp
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string fileName = $"{instrumentName}_{timestamp}.csv";
                csvFilePath = System.IO.Path.Combine(exportDir, fileName);
                
                // Check if file exists and is locked - delete if needed
                if (File.Exists(csvFilePath))
                {
                    try
                    {
                        // Try to delete the file if it exists
                        File.Delete(csvFilePath);
                    }
                    catch (Exception fileEx)
                    {
                        Print($"WARNING: Could not delete existing file: {fileEx.Message}");
                        // If we can't delete, use a new filename
                        timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                        fileName = $"{instrumentName}_{timestamp}.csv";
                        csvFilePath = System.IO.Path.Combine(exportDir, fileName);
                    }
                }
                
                // Create and initialize the CSV file with headers
                csvWriter = new StreamWriter(csvFilePath, false, Encoding.UTF8);
                
                // Write header row - using lowercase column names to match Python expectations
                csvWriter.WriteLine("instrument,datetime,open,high,low,close,ema_short,ema_long,atr,adx,rsi,bb_upper,bb_middle,bb_lower,macd,macd_signal,macd_hist,volume,date_value");
                csvWriter.Flush();
                
                Print($"Created CSV file: {csvFilePath}");
            }
            catch (Exception ex)
            {
                Print($"ERROR: Failed to prepare CSV file: {ex.Message}");
                csvWriter = null;
            }
        }
        
        /// <summary>
        /// Closes the CSV file and releases resources
        /// </summary>
        private void CloseCSVFile()
        {
            if (csvWriter != null)
            {
                try
                {
                    csvWriter.Flush();
                    csvWriter.Close();
                    csvWriter.Dispose();
                }
                catch (Exception ex)
                {
                    Print($"WARNING: Error closing CSV file: {ex.Message}");
                }
                finally
                {
                    csvWriter = null;
                }
            }
        }
        
        /// <summary>
        /// Process a batch of historical bars for extraction
        /// </summary>
        private void ProcessHistoricalBatch()
        {
            // Check if extraction is complete
            if (extractedBarsCount >= totalBarsToExtract || !isExtractingData || csvWriter == null)
            {
                CompleteExtraction();
                return;
            }
            
            // Verify that we have enough bars
            if (CurrentBar < 20)
            {
                Print("ERROR: Not enough bars available for extraction");
                CompleteExtraction();
                return;
            }
            
            // Set batch size for processing
            int batchSize = 25; // Process 25 bars at a time
            int remainingBars = totalBarsToExtract - extractedBarsCount;
            int barsToProcess = Math.Min(batchSize, remainingBars);
            int processedInBatch = 0;
            
            // Make sure we're not trying to extract more bars than we have
            int availableBars = CurrentBar + 1;
            if (totalBarsToExtract > availableBars)
            {
                Print($"WARNING: Requested {totalBarsToExtract} bars but only {availableBars} are available. Adjusting.");
                totalBarsToExtract = availableBars;
                remainingBars = totalBarsToExtract - extractedBarsCount;
                barsToProcess = Math.Min(batchSize, remainingBars);
            }
            
            // Safety check
            if (barsToProcess <= 0)
            {
                Print("No more bars to process. Completing extraction.");
                CompleteExtraction();
                return;
            }
            
            // Calculate starting bar index - we extract from oldest to newest
            int startBarsAgo = Math.Min(CurrentBar, totalBarsToExtract - extractedBarsCount);
            
            // Process each bar in this batch
            for (int i = 0; i < barsToProcess && (startBarsAgo - i) >= 0; i++)
            {
                int barsAgo = startBarsAgo - i;
                
                try
                {
                    // Format and write bar data
                    string barData = FormatBarDataForCSV(barsAgo);
                    if (!string.IsNullOrEmpty(barData))
                    {
                        csvWriter.WriteLine(barData);
                        processedInBatch++;
                    }
                }
                catch (Exception ex)
                {
                    Print($"Error processing bar with barsAgo={barsAgo}: {ex.Message}");
                }
            }
            
            // Ensure data is written to disk
            try
            {
                if (csvWriter != null)
                {
                    csvWriter.Flush();
                }
            }
            catch (Exception ex)
            {
                Print($"WARNING: Error flushing CSV data: {ex.Message}");
            }
            
            // Update progress
            extractedBarsCount += processedInBatch;
            progressPercent = (double)extractedBarsCount / totalBarsToExtract * 100;
            
            // Update status text and progress bar
            UpdateStatus($"Extracting: {extractedBarsCount}/{totalBarsToExtract}", progressPercent);
            
            Print($"Extraction progress: {progressPercent:F1}% ({extractedBarsCount}/{totalBarsToExtract}, processed {processedInBatch} bars in this batch)");
            
            // Check if complete
            if (extractedBarsCount >= totalBarsToExtract || processedInBatch == 0)
            {
                if (processedInBatch == 0 && extractedBarsCount < totalBarsToExtract)
                {
                    Print("WARNING: No bars processed in latest batch, completing extraction early.");
                }
                CompleteExtraction();
            }
            else
            {
                // Schedule the next batch with a short delay
                Thread.Sleep(10);
                try
                {
                    this.Dispatcher.InvokeAsync(() => ProcessHistoricalBatch());
                }
                catch (Exception ex)
                {
                    Print($"ERROR: Failed to schedule next batch: {ex.Message}");
                    CompleteExtraction();
                }
            }
        }
        
        /// <summary>
        /// Completes the extraction process
        /// </summary>
        private void CompleteExtraction()
        {
            if (!isExtractionComplete)
            {
                isExtractionComplete = true;
                isExtractingData = false;
                
                // Close the CSV file
                if (csvWriter != null)
                {
                    try
                    {
                        csvWriter.Flush();
                        csvWriter.Close();
                        csvWriter.Dispose();
                        
                        string instrumentName = Instrument.MasterInstrument.Name;
                        Print($"Historical data extraction completed for {instrumentName}: {extractedBarsCount} bars exported to {csvFilePath}");
                        
                        // Update UI with completion status
                        UpdateStatus($"Completed: {extractedBarsCount} bars", 100);
                    }
                    catch (Exception ex)
                    {
                        Print($"WARNING: Error closing CSV file: {ex.Message}");
                        UpdateStatus("Error completing extraction", progressPercent);
                    }
                    finally
                    {
                        csvWriter = null;
                    }
                }
                
                // Re-enable the extract button
                if (extractButton != null)
                {
                    ChartControl.Dispatcher.InvokeAsync(() => {
                        extractButton.IsEnabled = true;
                    });
                }
            }
        }
        
        /// <summary>
        /// Formats a single bar's data for CSV export
        /// </summary>
        private string FormatBarDataForCSV(int barsAgo)
        {
            try
            {
                // Double check bar index to prevent out of range errors
                if (barsAgo < 0 || barsAgo > CurrentBar)
                {
                    Print($"WARNING: Invalid barsAgo={barsAgo} (Current bar: {CurrentBar})");
                    return null;
                }
                
                // Safety check for accessing time data
                DateTime barTime;
                try
                {
                    barTime = Time[barsAgo];
                }
                catch (Exception ex)
                {
                    Print($"ERROR: Cannot access Time at index {barsAgo}: {ex.Message}");
                    return null;
                }
                
                // Get instrument name
                string instrumentName = Instrument.MasterInstrument.Name;
                
                // Format date/time
                string formattedDateTime = barTime.ToString("yyyy-MM-dd HH:mm:ss");
                double dateValue = barTime.ToOADate();
                
                // Get OHLC data with safety checks
                double openValue, highValue, lowValue, closeValue, volume;
                
                try
                {
                    openValue = Open[barsAgo];
                    highValue = High[barsAgo];
                    lowValue = Low[barsAgo];
                    closeValue = Close[barsAgo];
                    volume = Volume[barsAgo];
                }
                catch (Exception ex)
                {
                    Print($"ERROR: Cannot access price data at index {barsAgo}: {ex.Message}");
                    return null;
                }
                
                // Get indicator values - handle exceptions for each indicator separately
                double emaShortValue, emaLongValue, atrValue, adxValue, rsiValue, bbuValue, bbmValue, bblValue;
                double macdValue, macdSignalValue, macdHistValue;
                
                try { emaShortValue = emaShort != null && emaShort.IsValidDataPoint(barsAgo) ? emaShort[barsAgo] : closeValue; }
                catch (Exception) { emaShortValue = closeValue; }
                
                try { emaLongValue = emaLong != null && emaLong.IsValidDataPoint(barsAgo) ? emaLong[barsAgo] : closeValue * 0.98; }
                catch (Exception) { emaLongValue = closeValue * 0.98; }
                
                try { atrValue = atr != null && atr.IsValidDataPoint(barsAgo) ? atr[barsAgo] : (highValue - lowValue) * 0.5; }
                catch (Exception) { atrValue = (highValue - lowValue) * 0.5; }
                
                try { adxValue = adx != null && adx.IsValidDataPoint(barsAgo) ? adx[barsAgo] : 25.0; }
                catch (Exception) { adxValue = 25.0; }
                
                try { rsiValue = rsi != null && rsi.IsValidDataPoint(barsAgo) ? rsi[barsAgo] : 50.0; }
                catch (Exception) { rsiValue = 50.0; }
                
                try { bbuValue = bollinger != null && bollinger.Upper.IsValidDataPoint(barsAgo) ? bollinger.Upper[barsAgo] : closeValue * 1.02; }
                catch (Exception) { bbuValue = closeValue * 1.02; }
                
                try { bbmValue = bollinger != null && bollinger.Middle.IsValidDataPoint(barsAgo) ? bollinger.Middle[barsAgo] : closeValue; }
                catch (Exception) { bbmValue = closeValue; }
                
                try { bblValue = bollinger != null && bollinger.Lower.IsValidDataPoint(barsAgo) ? bollinger.Lower[barsAgo] : closeValue * 0.98; }
                catch (Exception) { bblValue = closeValue * 0.98; }
                
                try { macdValue = macd != null && macd.Default.IsValidDataPoint(barsAgo) ? macd.Default[barsAgo] : 0.0; }
                catch (Exception) { macdValue = 0.0; }
                
                try { macdSignalValue = macd != null && macd.Avg.IsValidDataPoint(barsAgo) ? macd.Avg[barsAgo] : 0.0; }
                catch (Exception) { macdSignalValue = 0.0; }
                
                try { macdHistValue = macd != null && macd.Diff.IsValidDataPoint(barsAgo) ? macd.Diff[barsAgo] : 0.0; }
                catch (Exception) { macdHistValue = 0.0; }
                
                // Format as CSV (no header) - using lowercase names to match Python expectations
                return string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}",
                    instrumentName,
                    formattedDateTime,
                    openValue.ToString("F2"),      // open
                    highValue.ToString("F2"),      // high
                    lowValue.ToString("F2"),       // low
                    closeValue.ToString("F2"),     // close
                    emaShortValue.ToString("F2"),  // ema_short
                    emaLongValue.ToString("F2"),   // ema_long
                    atrValue.ToString("F2"),       // atr
                    adxValue.ToString("F2"),       // adx
                    rsiValue.ToString("F2"),       // rsi
                    bbuValue.ToString("F2"),       // bb_upper
                    bbmValue.ToString("F2"),       // bb_middle
                    bblValue.ToString("F2"),       // bb_lower
                    macdValue.ToString("F4"),      // macd
                    macdSignalValue.ToString("F4"),// macd_signal
                    macdHistValue.ToString("F4"),  // macd_hist
                    volume.ToString("F0"),         // volume
                    dateValue.ToString());         // date_value
            }
            catch (Exception ex)
            {
                Print($"Error formatting bar data for CSV: {ex.Message}");
                return null;
            }
        }
        #endregion
        
        public override string ToString()
        {
            string status = " [";
            
            if (isExtractingData)
                status += $"Extracting: {extractedBarsCount}/{totalBarsToExtract}";
            else if (isExtractionComplete)
                status += "Extraction Complete";
            else
                status += "Ready";
                
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
        private RLLink[] cacheRLLink;
        public RLLink RLLink(string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            return RLLink(Input, exportFolderPath, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, maxHistoricalBars, showControlPanel);
        }

        public RLLink RLLink(ISeries<double> input, string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            if (cacheRLLink != null)
                for (int idx = 0; idx < cacheRLLink.Length; idx++)
                    if (cacheRLLink[idx] != null && cacheRLLink[idx].ExportFolderPath == exportFolderPath && cacheRLLink[idx].EMAShortPeriod == eMAShortPeriod && cacheRLLink[idx].EMALongPeriod == eMALongPeriod && cacheRLLink[idx].ATRPeriod == aTRPeriod && cacheRLLink[idx].ADXPeriod == aDXPeriod && cacheRLLink[idx].RSIPeriod == rSIPeriod && cacheRLLink[idx].BollingerPeriod == bollingerPeriod && cacheRLLink[idx].BollingerStdDev == bollingerStdDev && cacheRLLink[idx].MACDFast == mACDFast && cacheRLLink[idx].MACDSlow == mACDSlow && cacheRLLink[idx].MACDSignal == mACDSignal && cacheRLLink[idx].MaxHistoricalBars == maxHistoricalBars && cacheRLLink[idx].ShowControlPanel == showControlPanel && cacheRLLink[idx].EqualsInput(input))
                        return cacheRLLink[idx];
            return CacheIndicator<RLLink>(new RLLink(){ ExportFolderPath = exportFolderPath, EMAShortPeriod = eMAShortPeriod, EMALongPeriod = eMALongPeriod, ATRPeriod = aTRPeriod, ADXPeriod = aDXPeriod, RSIPeriod = rSIPeriod, BollingerPeriod = bollingerPeriod, BollingerStdDev = bollingerStdDev, MACDFast = mACDFast, MACDSlow = mACDSlow, MACDSignal = mACDSignal, MaxHistoricalBars = maxHistoricalBars, ShowControlPanel = showControlPanel }, input, ref cacheRLLink);
        }
    }
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
    public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
    {
        public Indicators.RLLink RLLink(string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            return indicator.RLLink(Input, exportFolderPath, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, maxHistoricalBars, showControlPanel);
        }

        public Indicators.RLLink RLLink(ISeries<double> input , string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            return indicator.RLLink(input, exportFolderPath, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, maxHistoricalBars, showControlPanel);
        }
    }
}

namespace NinjaTrader.NinjaScript.Strategies
{
    public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
    {
        public Indicators.RLLink RLLink(string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            return indicator.RLLink(Input, exportFolderPath, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, maxHistoricalBars, showControlPanel);
        }

        public Indicators.RLLink RLLink(ISeries<double> input , string exportFolderPath, int eMAShortPeriod, int eMALongPeriod, int aTRPeriod, int aDXPeriod, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int maxHistoricalBars, bool showControlPanel)
        {
            return indicator.RLLink(input, exportFolderPath, eMAShortPeriod, eMALongPeriod, aTRPeriod, aDXPeriod, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, maxHistoricalBars, showControlPanel);
        }
    }
}

#endregion