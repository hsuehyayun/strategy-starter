## Cryptocurrency Trading Bot Competition

- **Prize:** Nintendo Switch + Mario Kart
- **Goal:** build your own cryptocurrency trading bot and strategy using this repo as your launchpad
- **Live run:** organisers will deploy qualified bots over the Christmas holidays 2025
- **Judging:** weighted between strategy idea quality and the PnL generated during the live run
- **Deadline:** 31 December 2025 @ 23:59 UTC
- **Submission:** share your GitHub repository through the Google Form (link TBA)
- **Support:** weekly workshops plus the Building Division WhatsApp channel
- **Universe:** any Solana-based asset with an available price feed (template defaults to SOL but you may switch to any token)
- **AI tools:** use of AI assistants to understand or extend the codebase is not only allowed but encouraged

---

## Competition Flow

1. **Develop:** fork/clone this repo, craft your own indicators, execution logic, and risk management.
2. **Test:** backtest or paper trade; you may generate and fund your own wallet for testing (optional).
3. **Submit:** provide your GitHub repository and documentation before the deadline.
4. **Live run:** organisers fund approved wallets and run the bots over the holidays.
5. **Winner:** decided on both strategy rationale and realised PnL.

---

## Get Started with the Starter Repo

1. **Clone the starter repo**

   ```bash
   git clone https://github.com/GlenFilson/strategy-starter.git
   cd strategy-starter
   ```

2. **Install Node dependencies**

   ```bash
   npm install
   ```

3. **Create or import a Solana keypair (optional)**

   ```bash
   node generateWallet.js
   ```

   The script outputs `new-wallet.json`. Keep the secret safe—approved submissions are funded by the organisers prior to the live run. Only self-fund if you want to test privately.

4. **Create your `.env` file**

   ```
   PRIVATE_KEY=[JSON array copied from new-wallet.json]
   SOLANA_RPC=https://api.mainnet-beta.solana.com
   HELIUS_API_KEY=[your Helius RPC key]
   JUPITER_API=[your Jupiter quote/execution endpoint base URL]
   ```

5. **Run the reference bot**
   ```bash
   node bot.js
   ```
   The baseline bot streams Pyth prices, builds short/long SMAs, and trades on crossovers for SOL. Update the configuration (asset symbol, mint addresses, feeds) to target any Solana token with a supported price feed and expand the logic as needed.

---

## Repository Contents

- `bot.js` – real-time Solana trading scaffold with SMA example logic
- `generateWallet.js` – helper for creating Solana keypairs
- `fetch_pyth_data.py` + sample CSVs – tools for downloading historical OHLC data
- `backtest.ipynb` – notebook template for research and prototyping

Adapt or replace any component as long as your submission instructions explain how to run the final bot.

---

## Submission Checklist

- GitHub repository includes:
  - Live trading bot code and configuration
  - Brief README outlining strategy idea
  - Any additional resources that may support your submission
- Access granted to the judging team (public repo or invite required reviewers)
- Repository link submitted via Google Form (TBA) before the deadline
