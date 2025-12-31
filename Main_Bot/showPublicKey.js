import { Keypair } from "@solana/web3.js";
import dotenv from "dotenv";
dotenv.config();

const secret = JSON.parse(process.env.PRIVATE_KEY);
const keypair = Keypair.fromSecretKey(new Uint8Array(secret));
console.log("Your Public Key:", keypair.publicKey.toBase58());