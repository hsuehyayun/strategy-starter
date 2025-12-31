import WebSocket from "ws";
import Decimal from "decimal.js";
import {
  Connection,
  Keypair,
  PublicKey,
  VersionedTransaction,
  TransactionMessage,
  TransactionInstruction,
} from "@solana/web3.js";
import splToken from "@solana/spl-token";
const { getAssociatedTokenAddressSync } = splToken;
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

// ============================================================================
// CONFIGURATION - Strategy B (SMA + RSI + Fear & Greed)
// ============================================================================

const ASSET = "SOL";
const PYTH_ID =
  "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d";

const WS_URL = "wss://hermes.pyth.network/ws";
const DEBUG = true; // Set to true for verbose logging

// Candlestick settings
const CANDLESTICK_DURATION = 1000 * 60 * 60; // 1 hour in milliseconds
const CANDLESTICK_INTERVAL = "1h";
const SYMBOL = "SOLUSDT";
const CANDLESTICK_WINDOW_SIZE = 100; // Keep more candles for indicators

// ============================================================================
// STRATEGY B PARAMETERS (Optimized via backtest)
// ============================================================================

// SMA Parameters
const SMA_SHORT_PERIOD = 10;
const SMA_LONG_PERIOD = 50;

// RSI Parameters
const RSI_PERIOD = 14;
const RSI_OVERSOLD = 30;      // Buy only when RSI > 30
const RSI_OVERBOUGHT = 70;    // Buy only when RSI < 70
const RSI_EXIT_THRESHOLD = 75; // Exit when RSI > 75

// Fear & Greed Parameters
const FG_ENTRY_MAX = 75;      // Don't buy when F&G > 75 (too greedy)
const FG_EXIT_THRESHOLD = 80; // Exit when F&G > 80 (extreme greed)
const FG_UPDATE_INTERVAL = 1000 * 60 * 60; // Update F&G every hour

// Risk Management
const STOP_LOSS = 0.05;       // 5% stop loss
const TAKE_PROFIT = 0.15;     // 15% take profit
const TRADE_PERCENTAGE = 0.1; // 10% of portfolio per trade
const RESERVE_SOL_FOR_FEES = 0.02;
const SLIPPAGE_BPS = 50;
const JITO_TIP_LAMPORTS = 1000;

// ============================================================================
// INITIALIZATION
// ============================================================================

const ws = new WebSocket(WS_URL);

// Solana setup
const connection = new Connection(
  process.env.SOLANA_RPC || "https://api.mainnet-beta.solana.com"
);
const secret = JSON.parse(process.env.PRIVATE_KEY);
const keypair = Keypair.fromSecretKey(new Uint8Array(secret));

const SOL_MINT = new PublicKey("So11111111111111111111111111111111111111112");
const USDC_MINT = new PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");

// ============================================================================
// LOGGING UTILITY
// ============================================================================

function log(message) {
  console.log(`[${new Date().toISOString()}] ${message}`);
}

function debugLog(message) {
  if (DEBUG) {
    console.log(`[${new Date().toISOString()}] [DEBUG] ${message}`);
  }
}

// ============================================================================
// CANDLE DATA STRUCTURE
// ============================================================================

class Candle {
  constructor(timestamp, open, high, low, close) {
    this.timestamp = timestamp;
    this.open = open;
    this.high = high;
    this.low = low;
    this.close = close;
  }

  toString() {
    return `${new Date(this.timestamp).toISOString()} - O: ${this.open.toFixed(2)} H: ${this.high.toFixed(2)} L: ${this.low.toFixed(2)} C: ${this.close.toFixed(2)}`;
  }
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const candles = [];

const indicators = {
  smaShort: [],
  smaLong: [],
  rsi: [],
};

// Fear & Greed state
let fearGreedIndex = 50; // Default neutral
let lastFGUpdate = 0;

// Position tracking for stop loss / take profit
let position = {
  isOpen: false,
  entryPrice: 0,
  side: null, // 'LONG' or null
};

// Trade execution lock
let isExecutingTrade = false;

// ============================================================================
// FEAR & GREED INDEX FETCHER
// ============================================================================

async function fetchFearGreedIndex() {
  try {
    const response = await axios.get("https://api.alternative.me/fng/", {
      timeout: 10000,
    });
    
    if (response.data && response.data.data && response.data.data[0]) {
      const newFG = parseInt(response.data.data[0].value);
      fearGreedIndex = newFG;
      lastFGUpdate = Date.now();
      log(`Fear & Greed Index updated: ${newFG} (${response.data.data[0].value_classification})`);
      return newFG;
    }
  } catch (error) {
    log(`Failed to fetch Fear & Greed: ${error.message}`);
  }
  return fearGreedIndex; // Return cached value on error
}

// ============================================================================
// WEBSOCKET LIFECYCLE
// ============================================================================

ws.onopen = async () => {
  log("Connected to Pyth WebSocket");
  logStartupConfiguration();

  // Fetch initial Fear & Greed Index
  await fetchFearGreedIndex();

  // Fetch historical candles
  const startTime = Date.now() - CANDLESTICK_DURATION * CANDLESTICK_WINDOW_SIZE;
  const endTime = Date.now();

  log(`Fetching historical candles for ${SYMBOL} at ${CANDLESTICK_INTERVAL} interval`);

  await fetchHistoricalCandles(startTime, endTime, SYMBOL, CANDLESTICK_INTERVAL);
  log(`Fetched ${candles.length} candles`);

  updateIndicators();

  log(`🔔 Subscribing to ${ASSET} price updates...`);
  ws.send(
    JSON.stringify({
      type: "subscribe",
      ids: [PYTH_ID],
    })
  );
  log(`Subscribed to ${ASSET} price updates`);
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type !== "price_update") return;

  const { price, confidence, timestamp } = parsePrice(data.price_feed);
  onTick(price, timestamp);
};

ws.onerror = (error) => {
  log(`WebSocket error: ${error.message}`);
};

ws.onclose = () => {
  log("WebSocket disconnected");
};

// ============================================================================
// PRICE TICK HANDLER
// ============================================================================

async function onTick(price, timestamp) {
  const numericPrice = price.toNumber();
  
  // Update Fear & Greed periodically
  if (Date.now() - lastFGUpdate > FG_UPDATE_INTERVAL) {
    fetchFearGreedIndex(); // Don't await, run in background
  }

  // Check stop loss / take profit for open positions
  if (position.isOpen) {
    const pnlPercent = (numericPrice - position.entryPrice) / position.entryPrice;
    
    // Stop Loss
    if (pnlPercent <= -STOP_LOSS) {
      log(`STOP LOSS triggered at $${numericPrice.toFixed(2)} (${(pnlPercent * 100).toFixed(2)}%)`);
      if (!isExecutingTrade) {
        executeTrade({ signal: "SELL", price, reason: "STOP_LOSS" }, numericPrice);
      }
      return;
    }
    
    // Take Profit
    if (pnlPercent >= TAKE_PROFIT) {
      log(`TAKE PROFIT triggered at $${numericPrice.toFixed(2)} (${(pnlPercent * 100).toFixed(2)}%)`);
      if (!isExecutingTrade) {
        executeTrade({ signal: "SELL", price, reason: "TAKE_PROFIT" }, numericPrice);
      }
      return;
    }
  }

  // Update candlestick
  const candleClosed = updateCandles(price, timestamp);
  let signal = null;

  if (candleClosed) {
    updateIndicators(true);
    debugLog("🕯️ Candle closed, indicators updated");
    signal = generateSignal(numericPrice);
  } else {
    updateIndicators(false);
  }

  // Execute trade if signal exists
  if (signal && !isExecutingTrade) {
    executeTrade(signal, numericPrice);
  }
}

// ============================================================================
// INDICATOR CALCULATIONS
// ============================================================================

function updateIndicators(appendIndicators = false) {
  const smaShortVal = calculateSMA(candles, SMA_SHORT_PERIOD);
  const smaLongVal = calculateSMA(candles, SMA_LONG_PERIOD);
  const rsiVal = calculateRSI(candles, RSI_PERIOD);

  if (appendIndicators) {
    pushAndTrim(indicators.smaShort, smaShortVal);
    pushAndTrim(indicators.smaLong, smaLongVal);
    pushAndTrim(indicators.rsi, rsiVal);
  }

  debugLog(
    `SMA ${SMA_SHORT_PERIOD}: ${smaShortVal?.toFixed(2) || "N/A"} | ` +
    `SMA ${SMA_LONG_PERIOD}: ${smaLongVal?.toFixed(2) || "N/A"} | ` +
    `RSI: ${rsiVal?.toFixed(1) || "N/A"} | ` +
    `F&G: ${fearGreedIndex}`
  );
}

function calculateSMA(candles, period) {
  if (candles.length < period) return null;
  const relevantCandles = candles.slice(-period);
  const sum = relevantCandles.reduce((acc, candle) => acc + candle.close, 0);
  return sum / period;
}

function calculateRSI(candles, period) {
  if (candles.length < period + 1) return null;
  
  const relevantCandles = candles.slice(-(period + 1));
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i < relevantCandles.length; i++) {
    const change = relevantCandles[i].close - relevantCandles[i - 1].close;
    if (change > 0) {
      gains += change;
    } else {
      losses += Math.abs(change);
    }
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  return rsi;
}

function pushAndTrim(array, value) {
  array.push(value);
  if (array.length > CANDLESTICK_WINDOW_SIZE) array.shift();
}

// ============================================================================
// SIGNAL GENERATION - Strategy B
// ============================================================================

function generateSignal(price) {
  const lastShort = indicators.smaShort[indicators.smaShort.length - 1];
  const lastLong = indicators.smaLong[indicators.smaLong.length - 1];
  const lastRSI = indicators.rsi[indicators.rsi.length - 1];
  
  const prevShort = indicators.smaShort[indicators.smaShort.length - 2];
  const prevLong = indicators.smaLong[indicators.smaLong.length - 2];

  // Need all indicators
  if (lastShort == null || lastLong == null || lastRSI == null) {
    return null;
  }

  // =========================================
  // SELL CONDITIONS
  // =========================================
  
  if (position.isOpen) {
    // Condition 1: SMA Death Cross (Short crosses below Long)
    if (lastShort < lastLong && prevShort != null && prevLong != null && prevShort >= prevLong) {
      log(`SELL Signal: SMA Death Cross at $${price.toFixed(2)}`);
      return { signal: "SELL", price, reason: "SMA_CROSS" };
    }
    
    // Condition 2: RSI Overbought Exit
    if (lastRSI > RSI_EXIT_THRESHOLD) {
      log(`SELL Signal: RSI Overbought (${lastRSI.toFixed(1)}) at $${price.toFixed(2)}`);
      return { signal: "SELL", price, reason: "RSI_OVERBOUGHT" };
    }
    
    // Condition 3: Fear & Greed Extreme Greed Exit
    if (fearGreedIndex > FG_EXIT_THRESHOLD) {
      log(`SELL Signal: Extreme Greed (F&G: ${fearGreedIndex}) at $${price.toFixed(2)}`);
      return { signal: "SELL", price, reason: "FG_EXIT" };
    }
  }

  // =========================================
  // BUY CONDITIONS
  // =========================================
  
  if (!position.isOpen) {
    // Check SMA Golden Cross
    const smaGoldenCross = lastShort > lastLong && 
                           prevShort != null && prevLong != null && 
                           prevShort <= prevLong;
    
    if (!smaGoldenCross) {
      return null; // No golden cross, no buy signal
    }
    
    // Check RSI filter (not oversold and not overbought)
    const rsiOK = lastRSI > RSI_OVERSOLD && lastRSI < RSI_OVERBOUGHT;
    if (!rsiOK) {
      debugLog(`BUY blocked: RSI out of range (${lastRSI.toFixed(1)})`);
      return null;
    }
    
    // Check Fear & Greed filter (not too greedy)
    const fgOK = fearGreedIndex < FG_ENTRY_MAX;
    if (!fgOK) {
      debugLog(`BUY blocked: F&G too high (${fearGreedIndex})`);
      return null;
    }
    
    // All conditions met!
    log(`BUY Signal: Golden Cross + RSI ${lastRSI.toFixed(1)} + F&G ${fearGreedIndex} at $${price.toFixed(2)}`);
    return { signal: "BUY", price, reason: "STRATEGY_B" };
  }

  return null;
}

// ============================================================================
// PRICE PARSER
// ============================================================================

function parsePrice(price_feed) {
  const price = new Decimal(price_feed.price.price);
  const confidence = new Decimal(price_feed.price.conf);
  const exponent = new Decimal(price_feed.price.expo);
  const timestamp = new Date(price_feed.price.publish_time * 1000);
  const actual_price = price.times(Math.pow(10, exponent.toNumber()));
  const actual_confidence = confidence.times(Math.pow(10, exponent.toNumber()));
  return { price: actual_price, confidence: actual_confidence, timestamp };
}

// ============================================================================
// CANDLESTICK MANAGEMENT
// ============================================================================

function updateCandles(price, timestamp) {
  if (candles.length === 0) return false;

  const numericPrice = price.toNumber();
  const currentCandle = candles[candles.length - 1];
  const currentCandleEndTimestamp = currentCandle.timestamp + CANDLESTICK_DURATION;

  if (timestamp >= currentCandleEndTimestamp) {
    const newTimestamp = currentCandleEndTimestamp;
    const newCandle = new Candle(
      newTimestamp,
      numericPrice,
      numericPrice,
      numericPrice,
      numericPrice
    );

    candles.push(newCandle);

    if (candles.length > CANDLESTICK_WINDOW_SIZE) {
      candles.shift();
    }
    debugLog(`New candle created at ${new Date(newTimestamp).toISOString()}`);
    return true;
  } else {
    currentCandle.high = Math.max(currentCandle.high, numericPrice);
    currentCandle.low = Math.min(currentCandle.low, numericPrice);
    currentCandle.close = numericPrice;
    return false;
  }
}

async function fetchHistoricalCandles(startTime, endTime, symbol, candleStickInterval) {
  const binanceUrl = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${candleStickInterval}&startTime=${startTime}&endTime=${endTime}&limit=1000`;

  try {
    const response = await axios.get(binanceUrl);
    const klines = response.data;

    for (const kline of klines) {
      const timestamp = parseInt(kline[0], 10);
      const open = parseFloat(kline[1]);
      const high = parseFloat(kline[2]);
      const low = parseFloat(kline[3]);
      const close = parseFloat(kline[4]);
      const candle = new Candle(timestamp, open, high, low, close);
      candles.push(candle);
      updateIndicators(true);
    }

    log(`📊 Historical candles loaded: ${candles.length} candles`);
    if (candles.length > 0) {
      log(`   First: ${candles[0].toString()}`);
      log(`   Last:  ${candles[candles.length - 1].toString()}`);
    }
  } catch (error) {
    log(`Failed to fetch historical candles: ${error.message}`);
  }
}

// ============================================================================
// TRADE EXECUTION
// ============================================================================

async function getJupiterQuote(inputMint, outputMint, amount) {
  const quoteUrl = `https://lite-api.jup.ag/swap/v1/quote?onlyDirectRoutes=true&inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount}&slippageBps=${SLIPPAGE_BPS}`;
  const response = await axios.get(quoteUrl);
  return response.data;
}

async function getJupiterSwapInstructions(quote, userPublicKey) {
  const response = await axios.post(
    "https://lite-api.jup.ag/swap/v1/swap-instructions",
    {
      userPublicKey: userPublicKey.toString(),
      quoteResponse: quote,
      wrapAndUnwrapSol: true,
      prioritizationFeeLamports: JITO_TIP_LAMPORTS,
      dynamicComputeUnitLimit: true,
    }
  );
  return response.data;
}

function createTransactionInstruction(instructionData) {
  return new TransactionInstruction({
    programId: new PublicKey(instructionData.programId),
    keys: instructionData.accounts.map((acc) => ({
      pubkey: new PublicKey(acc.pubkey),
      isSigner: acc.isSigner,
      isWritable: acc.isWritable,
    })),
    data: Buffer.from(instructionData.data, "base64"),
  });
}

async function executeTrade(signal, price) {
  if (isExecutingTrade) return;
  isExecutingTrade = true;

  try {
    const portfolioValue = await getPortfolioValue(price);
    if (!portfolioValue) return;

    const tradeAmount = calculateTradeAmount(signal, portfolioValue, price);
    if (tradeAmount === null) return;

    const inputMint = signal.signal === "BUY" ? USDC_MINT.toBase58() : SOL_MINT.toBase58();
    const outputMint = signal.signal === "BUY" ? SOL_MINT.toBase58() : USDC_MINT.toBase58();

    const quote = await getJupiterQuote(inputMint, outputMint, tradeAmount);
    if (!quote) throw new Error("No quote available");

    const swapData = await getJupiterSwapInstructions(quote, keypair.publicKey);

    const instructions = [];
    if (swapData.computeBudgetInstructions) {
      swapData.computeBudgetInstructions.forEach((ix) =>
        instructions.push(createTransactionInstruction(ix))
      );
    }
    if (swapData.setupInstructions) {
      swapData.setupInstructions.forEach((ix) =>
        instructions.push(createTransactionInstruction(ix))
      );
    }
    if (swapData.swapInstruction) {
      instructions.push(createTransactionInstruction(swapData.swapInstruction));
    }
    if (swapData.cleanupInstruction) {
      instructions.push(createTransactionInstruction(swapData.cleanupInstruction));
    }

    if (instructions.length === 0) throw new Error("No valid instructions");

    const { blockhash } = await connection.getLatestBlockhash();
    const messageV0 = new TransactionMessage({
      payerKey: keypair.publicKey,
      recentBlockhash: blockhash,
      instructions: instructions,
    }).compileToV0Message();

    const transaction = new VersionedTransaction(messageV0);
    transaction.sign([keypair]);

    const txid = await connection.sendRawTransaction(transaction.serialize(), {
      skipPreflight: true,
      maxRetries: 2,
    });

    const confirmation = await connection.confirmTransaction(txid, "confirmed");
    if (confirmation.value.err) throw new Error("Transaction failed");

    // Update position state
    if (signal.signal === "BUY") {
      position.isOpen = true;
      position.entryPrice = price;
      position.side = "LONG";
      log(`BUY executed @ $${price.toFixed(2)} | Reason: ${signal.reason} | TX: ${txid}`);
    } else {
      const pnl = position.isOpen ? ((price - position.entryPrice) / position.entryPrice * 100).toFixed(2) : "N/A";
      position.isOpen = false;
      position.entryPrice = 0;
      position.side = null;
      log(`SELL executed @ $${price.toFixed(2)} | Reason: ${signal.reason} | PnL: ${pnl}% | TX: ${txid}`);
    }
  } catch (error) {
    log(`Trade failed: ${error}`);
  } finally {
    isExecutingTrade = false;
  }
}

async function getPortfolioValue(price) {
  try {
    const usdcAccount = getAssociatedTokenAddressSync(USDC_MINT, keypair.publicKey);
    const solBalance = await connection.getBalance(keypair.publicKey);
    const usdcBalance = await connection.getTokenAccountBalance(usdcAccount);

    const reserveLamports = RESERVE_SOL_FOR_FEES * 1e9;
    const availableSolBalance = Math.max(0, solBalance - reserveLamports);
    const totalSolBalance = availableSolBalance / 1e9;
    const solInUSDC = totalSolBalance * price;
    const totalPortfolioUSDC = solInUSDC + (usdcBalance.value.uiAmount || 0);

    log(`💰 Portfolio: ${totalSolBalance.toFixed(4)} SOL + ${(usdcBalance.value.uiAmount || 0).toFixed(2)} USDC = $${totalPortfolioUSDC.toFixed(2)}`);

    return {
      solBalance: totalSolBalance,
      usdcBalance: usdcBalance.value.uiAmount || 0,
      totalPortfolioUSDC,
      availableSolLamports: availableSolBalance,
    };
  } catch (error) {
    log(`❌ Failed to fetch portfolio value: ${error.message}`);
    return null;
  }
}

function calculateTradeAmount(signal, portfolio, price) {
  const tradeAmountUSDC = Math.floor(portfolio.totalPortfolioUSDC * TRADE_PERCENTAGE * 1e6);

  if (signal.signal === "BUY") {
    const availableUSDC = portfolio.usdcBalance * 1e6;
    if (tradeAmountUSDC > availableUSDC) {
      log(`⏸️ Skipped: Insufficient USDC (have ${(availableUSDC / 1e6).toFixed(2)}, need ${(tradeAmountUSDC / 1e6).toFixed(2)})`);
      return null;
    }
    return tradeAmountUSDC;
  } else {
    const solToSell = (portfolio.totalPortfolioUSDC * TRADE_PERCENTAGE) / price;
    const solToSellLamports = Math.floor(solToSell * 1e9);
    const maxSellableLamports = portfolio.availableSolLamports;

    if (solToSellLamports > maxSellableLamports || solToSellLamports <= 0) {
      log(`⏸️ Skipped: Insufficient SOL (have ${(maxSellableLamports / 1e9).toFixed(4)}, need ${(solToSellLamports / 1e9).toFixed(4)})`);
      return null;
    }

    return solToSellLamports;
  }
}

// ============================================================================
// STARTUP LOGGING
// ============================================================================

function logStartupConfiguration() {
  log("============================================================");
  log("TRADING BOT - Strategy B (SMA + RSI + Fear & Greed)");
  log("============================================================");
  log(`Asset: ${ASSET}`);
  log(`Candle Interval: ${CANDLESTICK_INTERVAL}`);
  log(`SMA: ${SMA_SHORT_PERIOD} / ${SMA_LONG_PERIOD}`);
  log(`RSI: Period ${RSI_PERIOD}, Range ${RSI_OVERSOLD}-${RSI_OVERBOUGHT}, Exit ${RSI_EXIT_THRESHOLD}`);
  log(`F&G: Entry Max ${FG_ENTRY_MAX}, Exit ${FG_EXIT_THRESHOLD}`);
  log(`Risk: Stop Loss ${STOP_LOSS * 100}%, Take Profit ${TAKE_PROFIT * 100}%`);
  log(`Trade Size: ${TRADE_PERCENTAGE * 100}% of portfolio`);
  log(`Wallet: ${keypair.publicKey.toBase58()}`);
  log("============================================================");
}