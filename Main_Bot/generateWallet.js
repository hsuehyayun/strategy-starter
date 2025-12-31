import fs from "fs";
import { Keypair } from "@solana/web3.js";

// Generate a new keypair
const wallet = Keypair.generate();

// Save the secret key to a file (optional)
fs.writeFileSync(
  "new-wallet.json",
  JSON.stringify(Array.from(wallet.secretKey))
);

console.log("New wallet generated!");
console.log("Public Key:", wallet.publicKey.toBase58());
console.log("Secret Key saved to new-wallet.json");
