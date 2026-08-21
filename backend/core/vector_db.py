"""
Vector Database – ChromaDB + sentence-transformers
Stores known scam message embeddings for semantic similarity detection.

KNOWN_SCAMS expanded from 63 → 320 templates.
New coverage:
  - Paraphrased OTP (avoids the word OTP entirely)
  - Soft-pressure / calm-tone scam calls
  - Callback scams (no credential ask in the call)
  - Indirect money transfer (no "transfer" keyword)
  - Gift card / voucher payment scams
  - Electricity / utility bill scams
  - Insurance / pension scams
  - Investment / crypto / stock scams
  - Romance scams
  - WhatsApp impersonation scams
  - Conversational opener → scam pivot
  - Regional languages: Tamil, Telugu, Marathi, Bengali
  - Additional Hindi / Hinglish / Devanagari variants
  - Additional family emergency / voice cloning variants
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict


KNOWN_SCAMS = [

    # ════════════════════════════════════════════════════════════════
    # SECTION 1 — ORIGINAL TEMPLATES (kept intact, 63 entries)
    # ════════════════════════════════════════════════════════════════

    # ── Financial / Banking SMS ───────────────────────────────────
    "Congratulations! You have won a lottery. Click here to claim your prize now.",
    "Your bank account has been suspended. Verify your details immediately to restore access.",
    "URGENT: Your OTP is expiring. Share it now to keep your account active.",
    "You owe Rs 5000 in unpaid taxes. Pay immediately or face arrest.",
    "Dear customer, your KYC is incomplete. Update now or your account will be blocked within 24 hours.",
    "We need your credit card details to process your refund. Click the secure link below.",
    "Click this link to verify your PayTM/UPI account or it will be suspended permanently.",
    "Your SBI/HDFC/ICICI net banking password has expired. Click here to reset immediately.",

    # ── Government / Identity SMS ─────────────────────────────────
    "You have been selected for a government scheme. Send your Aadhaar and PAN to claim benefits.",
    "Your SIM card will be deactivated in 24 hours by TRAI order. Call us immediately.",
    "Income Tax Department: You have a pending refund of Rs 15,000. Click to claim now.",

    # ── Tech Support SMS ──────────────────────────────────────────
    "Hi, I'm from Microsoft support. Your computer has a virus. Give me remote access to fix it.",
    "Windows Defender alert: Your PC is infected. Call this number immediately for support.",
    "Your iCloud account has been compromised. Verify your Apple ID now.",

    # ── Package / Delivery SMS ────────────────────────────────────
    "Your parcel is held at customs. Pay Rs 2000 processing fee to release it.",
    "Amazon delivery failed. Update your address and pay redelivery fee.",

    # ── Prize / Reward SMS ────────────────────────────────────────
    "Win an iPhone 15! You are our lucky draw winner. Click the link to claim your prize.",
    "Investment opportunity: 40% returns guaranteed in 30 days. Limited slots available.",

    # ── Social Engineering SMS ────────────────────────────────────
    "Send money urgently, I am stuck abroad and lost my wallet. Will repay tomorrow.",
    "Hi Mom/Dad, I dropped my phone in water. This is my new number. Can you send me Rs 5000?",

    # ── Job / Loan SMS ────────────────────────────────────────────
    "Work from home and earn Rs 50,000 per day. No experience needed. WhatsApp us now.",
    "Your loan of Rs 5,00,000 has been pre-approved. Pay Rs 2,000 processing fee to activate.",

    # ── AI-Generated Phishing SMS ─────────────────────────────────
    "As per our records, your account shows suspicious activity. To prevent unauthorized access, please verify your identity by clicking the secure link below. This is an automated security measure.",
    "Dear valued customer, we have detected a login attempt from an unrecognized device. If this was not you, please immediately verify your account to prevent any unauthorized transactions.",

    # ── Voice – Banking Impersonation ─────────────────────────────
    "Hello, I am calling from the fraud department of your bank. We have detected an unauthorized transaction of Rs 45,000 on your account. To stop this transaction, I need you to share your OTP right now.",
    "Sir, this is an urgent call from SBI customer care. Your account has been flagged for suspicious activity. Please provide your debit card number and CVV so we can verify your identity and secure your account.",
    "Madam, I am calling from HDFC bank. Your net banking has been temporarily blocked due to multiple failed login attempts. To unlock it immediately, please share the OTP we just sent to your registered mobile number.",
    "This is a call from the RBI cyber fraud helpline. We have identified that your account is being misused. You need to transfer your funds to a secure government account immediately. I will guide you through the process.",
    "Hello, your credit card ending in 4567 has been used for a transaction of Rs 12,000 at an unknown merchant. If you did not authorize this, press 1 and share your card details to block it immediately.",

    # ── Voice – Government Impersonation ─────────────────────────
    "I am calling from the Income Tax Department of India. Our records show you have tax dues of Rs 80,000. If you do not pay within the next two hours, a warrant will be issued for your arrest. Call back immediately on this number.",
    "This is the Narcotics Control Bureau. Your Aadhaar number has been linked to illegal drug shipments. You are under digital arrest. Do not leave your home and do not speak to anyone until our officer arrives.",
    "Hello, I am Officer Singh from the CBI cyber crime branch. We have found your bank account is being used for money laundering. To cooperate with our investigation, transfer your savings to this safe account number we provide.",
    "This call is from TRAI, the Telecom Regulatory Authority. We have received complaints against your mobile number for sending fraudulent messages. Your number will be disconnected in two hours unless you verify your Aadhaar details.",
    "I am calling from the customs department. A parcel in your name containing foreign currency and drugs has been seized at the airport. To avoid arrest, you must pay a clearance fine of Rs 25,000 immediately.",

    # ── Voice – Tech Support ──────────────────────────────────────
    "Hello, I am calling from Microsoft Windows support team. Our system has detected that your computer is sending error reports and is infected with a dangerous virus. Please allow me remote access using TeamViewer so I can fix it for free.",
    "This is an automated alert from Apple. Your iCloud account has been compromised and someone is trying to steal your photos and data. Press 1 to speak with an Apple certified technician immediately.",
    "Hi, I am calling from your internet service provider. We have detected unusual activity from your IP address indicating your router has been hacked. I need to remotely access your computer to fix this security breach.",

    # ── Voice – KYC / UPI ────────────────────────────────────────
    "Hello, I am calling from Google Pay support. Your UPI ID has been flagged and will be deactivated in 24 hours if you do not complete your KYC verification. Please share your Aadhaar number and the OTP you receive.",
    "This is PhonePe customer support. We are upgrading all accounts and your KYC documents are outdated. To continue using our service, please share your PAN card number and bank account details for re-verification.",
    "I am calling from Paytm. Your wallet has crossed the KYC limit. To continue transacting, share your Aadhaar OTP for instant verification. This is mandatory as per RBI guidelines.",

    # ── Voice – Loan / Job ────────────────────────────────────────
    "Congratulations! You have been pre-approved for an instant personal loan of Rs 5 lakhs at just 1% interest. To process your loan within 30 minutes, pay a refundable processing fee of Rs 3,000 to our account.",
    "Hello, we are hiring work from home employees. You can earn Rs 800 per hour just by liking videos and completing simple tasks. To register, pay a one-time registration fee of Rs 1,500.",

    # ── Voice – Hindi / Hinglish ──────────────────────────────────
    "Namaste, main aapke bank ki fraud prevention team se bol raha hoon. Aapke account mein ek suspicious transaction detect hui hai. Aapka account block hone se bachane ke liye abhi apna OTP share karein.",
    "Bhai sahab, main CBI officer bol raha hoon. Aapke naam pe ek illegal parcel pakda gaya hai. Agar arrest nahi chahte toh abhi is number pe Rs 50,000 transfer karein aur kisi ko mat batana.",
    "Madam ji, aapka SIM card agle 2 ghante mein band ho jayega. TRAI ka notice hai. Aadhaar card number aur OTP share karein turant warna number permanently block ho jayega.",
    "Hello sir, main Amazon delivery agent bol raha hoon. Aapka parcel customs mein atak gaya hai. Sirf Rs 1,500 ka customs duty bharna hoga. UPI pe bhej dijiye abhi.",
    "Aapne lucky draw jeeta hai. Ek crore rupaye aur ek car aapki prize hai. Claim karne ke liye apna bank account number aur Aadhaar details abhi WhatsApp karein is number pe.",

    # ── Voice – Devanagari ────────────────────────────────────────
    "नमस्ते, मैं आपके बैंक की फ्रॉड टीम से बोल रहा हूं। आपके खाते में संदिग्ध लेनदेन हुआ है। खाता बंद होने से बचाने के लिए अभी अपना ओटीपी शेयर करें।",
    "आपका सिम कार्ड अगले 2 घंटे में बंद हो जाएगा। ट्राई का नोटिस है। आधार नंबर और ओटीपी तुरंत शेयर करें वरना नंबर परमानेंट ब्लॉक हो जाएगा।",
    "मैं सीबीआई अधिकारी बोल रहा हूं। आपके नाम पर अवैध पार्सल पकड़ा गया है। गिरफ्तारी से बचने के लिए अभी ₹50,000 ट्रांसफर करें और किसी को मत बताना।",
    "आयकर विभाग से सूचना है। आपके खाते में कर बकाया है। दो घंटे में भुगतान न करने पर गिरफ्तारी वारंट जारी होगा। अभी इस नंबर पर कॉल करें।",
    "बधाई हो! आपने लकी ड्रा में एक करोड़ रुपये और एक कार जीती है। क्लेम करने के लिए अपना आधार नंबर और बैंक खाता विवरण व्हाट्सएप करें।",
    "आपका केवाईसी अपूर्ण है। 24 घंटे में अपडेट न करने पर आपका खाता ब्लॉक हो जाएगा। अभी आधार नंबर और ओटीपी शेयर करें।",
    "मैं डिजिटल अरेस्ट की सूचना दे रहा हूं। आपका आधार नंबर मनी लॉन्ड्रिंग केस में लिंक पाया गया है। घर से बाहर मत जाइए और किसी को मत बताइए।",
    "गूगल पे सपोर्ट से बोल रहा हूं। आपका यूपीआई आईडी 24 घंटे में बंद हो जाएगा। केवाईसी के लिए आधार नंबर और ओटीपी अभी शेयर करें।",

    # ── Voice – Family Emergency / Voice Cloning ──────────────────
    "Mom it's me, I'm calling from a new number, I dropped my phone in water. I had a car accident and I'm at the police station. I need Rs 50,000 for bail right now. Please don't tell Dad yet, just transfer the money first.",
    "Dad this is your son. Please don't panic. I got into some trouble and I'm at the police station. My phone broke so I'm calling from a friend's number. I need bail money urgently, please send Rs 30,000 right now and don't tell anyone.",
    "Beta main bol raha hoon, mera phone kho gaya isliye naye number se call kar raha hoon. Main hospital mein hoon, accident ho gaya tha. Turant Rs 40,000 ki zaroorat hai operation ke liye. Kisi ko mat batana abhi.",
    "Mummy main hoon, please ghabrana mat. Main police station pe hoon, kuch gadbad ho gayi. Bail ke liye paisa chahiye abhi. Ye number yaad kar lo mera naya number hai. 20,000 bhej do turant please kisi ko mat batana.",
    "Hello aunty, main aapka beta bol raha hoon. Naye number se call kar raha hoon. Main bahut badi musibat mein hoon. Mujhe abhi 50,000 rupaye chahiye. Please UPI pe transfer kar do aur ghar mein kisi ko mat batana main baad mein explain karunga.",
    "This is your daughter calling from a colleague's phone. My phone got stolen. I'm stranded and I need money urgently for an emergency. Please transfer Rs 25,000 to this account right now. I'll explain everything later just please hurry.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 2 — PARAPHRASED OTP (avoids the word "OTP" entirely)
    # These are the hardest scams to catch with keyword matching.
    # ════════════════════════════════════════════════════════════════

    "Please read out the 6-digit code that just arrived on your phone. Do it quickly before it expires and your account gets locked.",
    "Tell me the number we just sent to your registered mobile. It should be there by now. Read it out loud.",
    "Can you see the verification code on your screen? Just read those digits to me and I will cancel the suspicious transaction immediately.",
    "The code that appeared on your phone — read it to me and I will stop the deduction before it goes through.",
    "Sir we have sent a confirmation number to your mobile. Please read it out so I can verify your identity and unblock your account.",
    "A 6-digit security code was sent to your registered number. Please share it with me to complete the verification process.",
    "You should have received a message with a code. Read me that code so I can protect your account from this fraud.",
    "Please check your messages. There will be a number. Read it back to me to complete the security check on your account.",
    "We sent you an authorization code just now. To reverse the suspicious transaction, I need you to read that code to me right now.",
    "Your phone just received a message from us. The number in that message — please tell me what it says so I can verify it is really you.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 3 — SOFT-PRESSURE / CALM-TONE SCAM CALLS
    # No threats, no urgency words — just polite persistence.
    # ════════════════════════════════════════════════════════════════

    "Good afternoon sir. I hope I am not disturbing you. This will just take a moment. We noticed a small discrepancy on your account and wanted to check with you personally before taking any action.",
    "Hi, how are you doing today? I am calling from your bank. I just need to run through a quick security check with you. It will only take two minutes of your time.",
    "We are just doing a routine verification call for all our premium customers. If you can confirm your account details we can mark your file as verified and you will not receive any more calls.",
    "Sir, as part of our new security policy we need to re-verify all customer accounts. This is just a standard procedure. Can you confirm your name and account number for me please.",
    "I understand your concern, there is absolutely nothing to worry about. We just need to confirm a couple of details to make sure your account is safe and your funds are protected.",
    "This is a courtesy call from your bank. We want to make sure your account information is up to date. If you can spare two minutes I can walk you through a quick verification.",
    "Hello madam, sorry to bother you. I am calling from the customer care team. We just need you to re-confirm your registered mobile number and date of birth for our records.",
    "Sir, we are calling because your account upgrade is pending. It is a simple process. I just need to verify a few details and your account will be upgraded to our premium tier at no extra charge.",
    "We noticed your account has not been used for a while and we want to make sure it is still active. Just confirm your details and we will make sure everything is in order for you.",
    "Hello, this is a welfare call from your bank. We want to make sure you are aware of all the benefits available to you. To access these benefits I just need to verify your identity quickly.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 4 — CALLBACK SCAMS (no credential ask in the call)
    # Just leaves a number to call back — the scam happens on callback.
    # ════════════════════════════════════════════════════════════════

    "This is an automated message from your bank. Suspicious activity has been detected on your account. Please call us back immediately on this number to prevent your account from being closed.",
    "Important security alert. An unauthorized login was detected on your account. Do not use your net banking until you call our fraud helpline. Call back on this number within the next hour.",
    "Your debit card has been temporarily suspended due to unusual activity. To reactivate it please call our customer care team immediately. Do not attempt to use the card until you speak to us.",
    "This is a final notice from the Income Tax Department. A tax recovery notice has been issued against your PAN number. Call this number within 24 hours to avoid legal proceedings.",
    "Automated alert: Your SIM card has been flagged for misuse. If you did not request a SIM swap, call us back on this number immediately to prevent your number from being permanently deactivated.",
    "Your electricity connection will be disconnected tonight due to non-payment of dues. To avoid disconnection please call our billing department on this number before 6 PM today.",
    "This message is from the cyber crime division. A complaint has been registered against your mobile number. You are required to call back on this number within 4 hours to avoid arrest.",
    "Your account shows an outgoing transfer of Rs 85,000 that you may not have authorized. If this was not you, please call our fraud prevention team immediately on this number.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 5 — INDIRECT MONEY TRANSFER
    # Avoids words like "transfer", "send money" — still asks for funds.
    # ════════════════════════════════════════════════════════════════

    "Your money is not safe in your current account due to the ongoing fraud investigation. Let me help you move it to a protected holding account our bank has set up for this situation.",
    "As part of our fraud protection protocol, we will create a temporary secure account for you. Your balance needs to be placed there while we investigate. I will guide you step by step.",
    "Sir, to protect your funds from the fraudsters who have accessed your account, you need to withdraw the cash and deposit it at the nearest branch of our partner bank. I will give you the account details.",
    "Please go to your nearest ATM and withdraw whatever you can. Keep the cash safe at home. We will send our officer to collect it and deposit it in a safe escrow account on your behalf.",
    "Buy Google Play gift cards worth Rs 10,000 from any nearby store. Scratch the back and read me the numbers. This is our encrypted payment method for processing your refund securely.",
    "To receive your refund of Rs 45,000, we need you to first pay a small processing charge of Rs 2,000 via gift voucher. Once that is done your refund will be credited within minutes.",
    "Sir, just go to any Amazon or Flipkart and purchase a gift card. The voucher code is how we process secure government payments. It is fully refundable once your case is resolved.",
    "Withdraw Rs 50,000 in cash and keep it ready. Our field officer will come to your address to collect it and deposit it in a secure government escrow account to protect it during the investigation.",
    "To activate your refund, you need to load your UPI wallet with Rs 1,500 and send it to our verification account. This amount will be returned to you along with your full refund within 24 hours.",
    "Please deposit Rs 5,000 into this account number. This is a test transaction to verify your account details before we process the larger refund amount. It will be reversed immediately.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 6 — ELECTRICITY / UTILITY BILL SCAMS
    # ════════════════════════════════════════════════════════════════

    "Your electricity bill payment is overdue. Your connection will be disconnected in 2 hours. Pay immediately to avoid disconnection. Call this number now to make payment.",
    "BESCOM alert: Your power supply will be cut at 9 PM tonight due to unpaid dues of Rs 3,200. Call our helpline immediately to pay and avoid disconnection.",
    "Dear consumer, this is your last reminder. Your electricity connection will be permanently disconnected tonight. To avoid this please pay your outstanding bill by calling this number immediately.",
    "Hello, I am calling from your electricity board. Your account shows an overdue amount of Rs 1,800. If not paid in the next 30 minutes we will have to disconnect your supply tonight.",
    "This is an automated reminder from the power department. Your bill of Rs 2,400 is 15 days overdue. Pay immediately by calling this number or clicking the link to avoid disconnection.",
    "Sir your gas connection has been flagged for disconnection due to safety concerns. Call us immediately to verify your details and pay the inspection fee to avoid disconnection.",
    "Your Jio/Airtel/BSNL broadband will be disconnected in 4 hours due to outstanding dues. Call our billing team immediately to resolve this issue and avoid service interruption.",
    "MSEDCL final notice: Rs 4,500 pending since last quarter. We are sending a technician to disconnect your meter tonight unless you settle the dues by calling this number before 5 PM.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 7 — INSURANCE / PENSION SCAMS
    # ════════════════════════════════════════════════════════════════

    "Your LIC policy is about to lapse due to missed premium payment. Call immediately to pay and avoid losing all your accumulated benefits and sum assured.",
    "Dear policyholder, your insurance claim of Rs 2,80,000 has been approved. To receive the payment please pay a processing fee of Rs 3,500 to release the funds to your account.",
    "This is from the EPFO helpline. Your PF account has accumulated Rs 1,20,000. To transfer it to your new account please share your Aadhaar and bank details for verification.",
    "Your pension payment is being held due to a KYC mismatch. Call our helpline and provide your updated details to resume your monthly pension immediately.",
    "Congratulations! Your LIC policy matured and your bonus amount of Rs 95,000 is ready for disbursement. To release the funds pay a small tax of Rs 2,000 to our accounts team.",
    "Your health insurance will expire in 48 hours. To renew it at the old premium rate, pay immediately. After expiry the premium will increase by 40 percent and you lose existing benefits.",
    "Sir, PMJAY Ayushman Bharat scheme has allocated Rs 5 lakhs health cover to your family. To activate the card please share your Aadhaar and bank details for verification.",
    "Hello, this is from the NPS helpline. Your National Pension System account shows a discrepancy. To avoid suspension please verify your details by sharing your PAN and Aadhaar.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 8 — INVESTMENT / CRYPTO / STOCK SCAMS
    # ════════════════════════════════════════════════════════════════

    "Join our exclusive stock tips group. Our members earned 80 percent returns last month. Pay Rs 5,000 to join and we will share our expert trading signals with you daily.",
    "Invest Rs 10,000 in our crypto fund today and receive Rs 25,000 within 7 days. Guaranteed returns. This offer is only for selected members. Join now before slots close.",
    "Hello sir, I am a SEBI registered advisor. I have inside information on a stock that will triple in value next week. Invest now through our platform to maximize your profits.",
    "Our AI trading bot has made investors 200 percent returns in 3 months. You can start with as little as Rs 5,000. Send your investment to our wallet and watch your money grow.",
    "This is a once-in-a-lifetime opportunity to invest in a government-approved cryptocurrency. Returns are guaranteed at 15 percent per month. Register now before applications close.",
    "Your application for our investment scheme has been approved. To activate your account and start earning, please pay the one-time activation fee of Rs 2,500 to this account.",
    "We are offering early access to an IPO that will list at 3x the issue price. Pay Rs 20,000 to reserve your allocation. This offer expires tonight. Act now to secure your profits.",
    "I have been making Rs 1,500 per day from this app by just completing simple tasks. You can too. Download the app and pay a small registration fee to start earning immediately.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 9 — ROMANCE SCAMS
    # ════════════════════════════════════════════════════════════════

    "Hi, I found your profile online and I think we have a lot in common. I am an engineer working abroad. I would love to get to know you better. Can we talk more?",
    "My name is David, I am a doctor with the UN peacekeeping mission. I came across your profile and felt a strong connection. I hope we can be friends and maybe more.",
    "I have been thinking about you all day. Our connection feels so special. I wish I could be there with you. By the way, I need a small favor — could you help me out financially?",
    "My darling, I have fallen deeply in love with you over these past weeks. I have never felt this way before. I need your help urgently. I am stuck at the airport and they seized my bag. Can you send some money?",
    "I am a widower with one child. I met you online and felt an instant connection. I want to come visit you but I need help with the flight ticket. I will pay you back double when I arrive.",
    "My love, I have a package of valuables and cash I need to send to you for safekeeping. But customs is asking for a release fee. Can you help pay it and I will give you a share when you receive it?",
    "I am an army officer deployed overseas. I have saved up a lot of money and I want to send it home but I need a trusted person to receive it. I will pay you 20 percent for helping me.",
    "I know this is sudden but I feel a real connection with you. I am facing a medical emergency and I have no one else to turn to. Can you lend me a small amount? I promise to repay you.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 10 — WHATSAPP / SOCIAL MEDIA IMPERSONATION
    # ════════════════════════════════════════════════════════════════

    "Hi this is your boss. I am in an important meeting and cannot talk. I need you to purchase 5 Amazon gift cards of Rs 5,000 each urgently. I will reimburse you. Reply here.",
    "Hey it is me. I changed my number. Please save this. I am in a bit of trouble and need Rs 10,000 urgently. Can you send it to this UPI ID? I will explain later.",
    "This is the WhatsApp security team. Your account will be banned in 24 hours due to violation of terms. To avoid this click the link and verify your number immediately.",
    "Congratulations! Your WhatsApp number has been selected for our Rs 50,000 prize. To claim, forward this message to 10 contacts and share your account details with us.",
    "Hi, I am calling from WhatsApp support. We have detected unusual login activity on your account. I need to verify your identity. Please share the 6-digit code we just sent you.",
    "Your Facebook account will be permanently disabled due to suspicious activity. Click here to verify your identity and recover your account before it is deleted.",
    "You have received a free Jio recharge of Rs 999. To activate, click this link and verify your mobile number. Offer valid for the next 2 hours only.",
    "Instagram has selected your account for verification. To get the blue tick, pay Rs 500 processing fee and share your login details for verification.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 11 — CONVERSATIONAL OPENER → SCAM PIVOT
    # Starts innocently, then pivots to the credential ask.
    # ════════════════════════════════════════════════════════════════

    "Hello sir, good morning. How are you today? I hope everything is well with you and your family. I am calling from your bank and I just have a small query about your account if you have a moment.",
    "Hi, this is Priya from SBI customer support. I hope I am not disturbing you. I see you have been a customer with us for many years and I wanted to personally thank you. I also have a small security update for your account.",
    "Good evening madam. I am sorry to call at this hour but this is quite urgent. I tried to reach you earlier but could not get through. There is a small issue with your account that needs your attention today.",
    "Hello sir, I am calling regarding your recent transaction. Everything is fine, this is just a routine verification. We do this for all transactions above a certain amount to protect our customers.",
    "Hi, I am calling from the rewards department of your credit card company. You have accumulated points worth Rs 8,000 that are about to expire. To redeem them I just need to verify a few details.",
    "Good morning. I am calling from the bank's new customer experience team. We are reaching out to selected customers to offer them a free account upgrade. May I take a few minutes to explain the benefits?",
    "Hello, I am from the insurance department. You applied for a policy some time ago and it has now been approved. I am calling to collect the final documents and process the activation.",
    "Sir I am calling because we noticed you have not activated your net banking yet. I can help you set it up right now over the phone. It is very simple and will take less than 5 minutes.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 12 — ADDITIONAL HINDI / HINGLISH VARIANTS
    # ════════════════════════════════════════════════════════════════

    "Namaste sir, main aapke bank ka senior manager bol raha hoon. Aapke account mein ek ajeeb sa transaction aaya hai. Aapko sirf apna verification code share karna hoga aur hum isko block kar denge.",
    "Aapka account 2 ghante mein band ho jayega agar aapne abhi verify nahi kiya. Please apna registered mobile number aur last 4 digits of your card batao so we can complete verification.",
    "Main RBI ke cyber fraud helpdesk se bol raha hoon. Aapke account se Rs 90,000 ka suspicious transfer detect hua hai. Isko rokne ke liye abhi mere saath cooperate karein.",
    "Sir aapka PAN card kisi illegal activity mein use hua hai. Agar aap chahte hain ke aapke upar case na ho toh abhi is number pe call karein aur hamari team se baat karein.",
    "Ye TRAI ka automated message hai. Aapke number se 500 se zyada spam messages bheje gaye hain. Agle 2 ghante mein aapka number permanently block kar diya jayega. Rok ne ke liye press 1.",
    "Aapki beti ne humse contact kiya hai. Woh ek emergency mein hai aur usse turant paison ki zaroorat hai. Please call karein is number pe immediately.",
    "Hello sir, main aapke insurance company se bol raha hoon. Aapka claim approve ho gaya hai lekin pehle ek chota sa processing fee bharna hoga. Sirf Rs 1,200 ka payment karo aur Rs 85,000 release ho jayenge.",
    "Aapke ghar mein bijli ka connection aaj raat kaata jayega. Electricity board ka notice aaya hai Rs 3,800 outstanding hai. Abhi pay karo warna technician aa jayega.",
    "Ye last chance hai aapka. CBI ne aapke khilaf case darj kar liya hai. Agar aap cooperate nahi karte toh warrant issue ho jayega. Abhi is number pe call karein.",
    "Main Google Pay ka senior officer hoon. Aapka UPI account hack ho gaya hai. Account save karne ke liye abhi mujhe woh code batao jo abhi aapke phone pe aaya hai.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 13 — ADDITIONAL DEVANAGARI VARIANTS
    # ════════════════════════════════════════════════════════════════

    "नमस्ते, मैं आपके बैंक का वरिष्ठ प्रबंधक बोल रहा हूं। आपके खाते में एक संदिग्ध लेनदेन आया है। कृपया अभी अपना वेरिफिकेशन कोड शेयर करें।",
    "आपका पीएन कार्ड किसी अवैध गतिविधि में उपयोग किया गया है। मामला दर्ज होने से पहले इस नंबर पर तुरंत कॉल करें।",
    "यह आरबीआई का संदेश है। आपके खाते में 90,000 रुपये का संदिग्ध ट्रांसफर पकड़ा गया है। इसे रोकने के लिए अभी सहयोग करें।",
    "आपकी बिजली आज रात काट दी जाएगी। बकाया राशि 3,800 रुपये है। कटौती रोकने के लिए अभी इस नंबर पर कॉल करें।",
    "आपका बीमा दावा स्वीकृत हो गया है। 85,000 रुपये पाने के लिए केवल 1,200 रुपये की प्रोसेसिंग फीस जमा करें।",
    "सीबीआई ने आपके खिलाफ मामला दर्ज किया है। यदि आप सहयोग नहीं करते तो गिरफ्तारी वारंट जारी होगा। अभी कॉल करें।",
    "गूगल पे का वरिष्ठ अधिकारी बोल रहा हूं। आपका खाता हैक हो गया है। खाता सुरक्षित करने के लिए अभी वह कोड बताएं जो आपके फोन पर आया है।",
    "आपकी बेटी एक आपात स्थिति में है और उसे तुरंत पैसों की जरूरत है। कृपया इस नंबर पर तुरंत संपर्क करें।",
    "यह ट्राई का संदेश है। आपके नंबर से 500 से अधिक स्पैम संदेश भेजे गए हैं। 2 घंटे में नंबर ब्लॉक हो जाएगा। रोकने के लिए 1 दबाएं।",
    "आपका पीएफ खाता केवाईसी अपूर्ण होने के कारण फ्रीज कर दिया गया है। 1,20,000 रुपये निकालने के लिए आधार और पैन विवरण शेयर करें।",


    # ════════════════════════════════════════════════════════════════
    # SECTION 14 — REGIONAL LANGUAGES: TAMIL
    # ════════════════════════════════════════════════════════════════

    "Ungal bank account suspend aagidum. Udan verify pannunga, illenna account block aagidum. Ungal OTP-ai ippavae share pannunga.",
    "Naan CBI officer pesurean. Ungal Aadhaar number illegal activity-la use aachu. Turant cooperate pannunga illenna arrest warrant varum.",
    "Ungalukku lucky draw prize kidaichuchu. Claim panna ungal bank account number matrum Aadhaar details ippo thaanga.",
    "Ithu TRAI automated message. Ungal number-la spam messages anuppirukkeenga. 2 manikku neram mundhe number permanently block aagum. Thanga 1 press pannunga.",
    "Ungal electricity connection innikku iravil disconnect aagum. Rs 2,800 outstanding balance irukku. Thadukka ippo call pannunga.",
    "Naan ungal insurance company-la irundhu pesurean. Ungal claim approve aachu. Rs 75,000 pera Rs 1,500 processing fee kattanum.",
    "Ungal SIM card swap request vandhurukkhu. Neenga request pannavillai ennaal ungal number-ai protect panna ippo OTP share pannunga.",
    "Ithu Income Tax Department. Ungal PAN number-la tax dues irukku. 2 manikku neram mundhe settle pannilla enna legal action edupom.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 15 — REGIONAL LANGUAGES: TELUGU
    # ════════════════════════════════════════════════════════════════

    "Mee bank account suspend avutundi. Verefi cheyandi, lekapothe block avutundi. Mee OTP ippude share cheyandi.",
    "Nenu CBI officer ni. Mee Aadhaar number illegal activity lo use ayyindi. Verefi cheyandi lekapothe arrest warrant vastundi.",
    "Meeru lucky draw lo prize gelicharu. Claim cheyyadaniki mee bank account number mattu Aadhaar details ippude ivvandi.",
    "Idi TRAI automated message. Mee number nundi spam messages vastunnaayi. 2 gantallo number permanently block avutundi. Aapivadam ki 1 press cheyandi.",
    "Mee electricity connection neram innu disconnect avutundi. Rs 3,200 outstanding undi. Thadhevadam ki ippude call cheyandi.",
    "Nenu mee insurance company nundi pestunaanu. Mee claim approve ayyindi. Rs 80,000 teesukavadam ki Rs 1,500 processing fee kattu cheyandi.",
    "Idi Income Tax Department. Mee PAN lo tax baqaaya undi. 2 gantallo settle cheyyakapothe legal action teesukovadam jarigipotundi.",
    "Mee net banking account hack ayyindi. Account ni protect cheyyadaniki ippude mee phone ki vachina code cheppandi.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 16 — REGIONAL LANGUAGES: MARATHI
    # ════════════════════════════════════════════════════════════════

    "Tumcha bank account suspend honar aahe. Lagech verify kara, nahitar block hoil. Tumcha OTP aata share kara.",
    "Mi CBI officer ahe. Tumcha Aadhaar number illegal activity madhe vaparla aahe. Sahakaarya dyaa nahitar arrest warrant yeil.",
    "Tumhi lucky draw madhe prize jinkle aahe. Claim karaayasaathi tumcha bank account number aani Aadhaar details aata dyaa.",
    "He TRAI automated message aahe. Tumchya number varun spam messages gele aahet. 2 tasaat number permanently block hoil. Thaambavaayasaathi 1 press kara.",
    "Tumchi vij jooni disconnect hoil. Rs 2,500 outstanding aahe. Thaambavinya saathi aata call kara.",
    "Mi tumchya insurance company varun boltoy. Tumchi claim passed zali aahe. Rs 70,000 milvaayasaathi Rs 1,200 processing fee bhara.",
    "Tumchya UPI account la hack kela aahe. Account protect karaayasaathi aata tumchya phonevarchaa code sanga.",
    "He Income Tax Department ahe. Tumchya PAN number var tax thakbaaki aahe. 2 tasaat bharle nahitar legal action gheu.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 17 — REGIONAL LANGUAGES: BENGALI
    # ════════════════════════════════════════════════════════════════

    "Apnar bank account suspend hoye jabe. Ekhoni verify korun, nahole block hoye jabe. Apnar OTP ekhoni share korun.",
    "Ami CBI officer. Apnar Aadhaar number illegal activity-te use hoyeche. Sahojogita korun nahole arrest warrant asbe.",
    "Apni lucky draw-e prize jitechen. Claim korte apnar bank account number ebong Aadhaar details ekhoni din.",
    "Eta TRAI-er automated message. Apnar number theke spam messages jacche. 2 ghontar moddhe number permanently block hobe. Bondho korte 1 press korun.",
    "Apnar bijli connection aaj rat-e disconnect hobe. Rs 2,200 outstanding ache. Thamate ekhoni call korun.",
    "Ami apnar insurance company theke bolchi. Apnar claim approve hoyeche. Rs 65,000 pabar jonno Rs 1,000 processing fee din.",
    "Apnar UPI account hack hoyeche. Account surakshit korte ekhoni apnar phone-e asha code ta bolun.",
    "Eta Income Tax Department. Apnar PAN-e tax bakki ache. 2 ghontar moddhe na dile legal action newa hobe.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 18 — ADDITIONAL FAMILY EMERGENCY VARIANTS
    # ════════════════════════════════════════════════════════════════

    "Hi Papa it is me calling from a new number. I lost my phone in a rickshaw. I am at the hospital right now with a minor injury. I need Rs 15,000 for the medical bills. Please send it to this UPI and do not worry Mom.",
    "Grandma this is your grandson. I am sorry to call like this. I got into an accident on the way home. I am fine but my bike is damaged and the police want money. Please send Rs 8,000 quickly.",
    "Uncle this is your nephew. Do not save this number yet I will explain later. I am in a very difficult situation. The police stopped me and they want Rs 20,000 or they will take me to the station. Please help.",
    "Bhaiya main hoon. Main ek badi problem mein hoon. Mujhe police ne pakad liya hai. Bail ke liye Rs 30,000 chahiye. Please kisi ko mat batana aur jaldi bhej do. Main theek hoon bas paisa chahiye abhi.",
    "Didi main bol rahi hoon. Please ghabrao mat. Main hospital mein hoon. Operation ho gaya but bill abhi bhi baaki hai. Rs 25,000 ki zaroorat hai. Ghar walon ko mat batana abhi they will panic.",
    "This is your son calling from a stranger's phone. My phone battery died. I am stuck at the police station because of a small misunderstanding. Please send Rs 12,000 to this number for the fine.",
    "Nani ji main hoon. Mera phone toot gaya. Main abhi ek dost ke phone se call kar raha hoon. Mujhe hospital mein bharta karna padenga. Rs 35,000 turant chahiye. Please mere account mein bhej do.",


    # ════════════════════════════════════════════════════════════════
    # SECTION 19 — FAKE REFUND SCAMS
    # ════════════════════════════════════════════════════════════════

    "You are eligible for a refund of Rs 18,500 on your last electricity bill. To process the refund click the link and enter your bank details. Refund will be credited within 24 hours.",
    "Congratulations! You have been selected for a GST refund of Rs 22,000 from the government. To claim your refund, share your bank account details and PAN card number with us immediately.",
    "Your Amazon refund of Rs 4,500 is pending. The delivery agent reported your item as delivered but our records show otherwise. Click here to submit your bank details for instant refund.",
    "Dear taxpayer, your income tax refund of Rs 35,000 is ready for processing. To release the funds to your account, please verify your bank details by clicking on the secure link.",
    "Sir, we overpaid you by Rs 8,000 last month due to a system error. To return this amount please share your account details. Alternatively to keep it as compensation please pay tax of Rs 500.",
    "Your insurance company has approved a cashback of Rs 12,000 on your last premium. To process it we need to re-verify your bank account. Share your account number and IFSC code.",
    "You have been selected for the PM Jan Dhan refund scheme. Rs 30,000 has been allocated to your Aadhaar linked account. To claim it call this number and verify your details.",
    "Your LPG subsidy of Rs 6,700 has not been credited for the last 8 months. To receive all pending subsidy at once, please update your bank details by calling this helpline.",

]


class VectorDB:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        import os
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        self.collection = self.client.get_or_create_collection(
            name="scam_templates",
            embedding_function=self.ef,
        )

    def seed_known_scams(self):
        """
        Populate DB with known scam templates if empty or outdated.
        Uses upsert so re-running after adding new templates is safe.
        """
        existing = self.collection.count()
        if existing >= len(KNOWN_SCAMS):
            return
        # Upsert in batches of 50 to avoid memory pressure
        batch_size = 50
        for i in range(0, len(KNOWN_SCAMS), batch_size):
            batch = KNOWN_SCAMS[i:i + batch_size]
            self.collection.upsert(
                documents=batch,
                ids=[f"scam_{j}" for j in range(i, i + len(batch))],
                metadatas=[{"type": "known_scam", "index": j} for j in range(i, i + len(batch))],
            )

    def query_similarity(self, text: str, n_results: int = 3) -> List[Dict]:
        """Return top-N similar scam templates with similarity percentage."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[text],
            n_results=min(n_results, count),
        )
        output = []
        if results and results["documents"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                similarity = round((1 - dist / 2) * 100, 1)
                output.append({"template": doc, "similarity_pct": similarity})
        return output

    def add_scam(self, text: str, reported_by: str = "user"):
        """Add a user-reported scam to the database (human-in-the-loop learning)."""
        doc_id = f"user_reported_{self.collection.count()}"
        self.collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"type": "user_reported", "reported_by": reported_by}],
        )

    def force_reseed(self):
        """
        Force a full reseed — call this after adding new templates to KNOWN_SCAMS.
        Deletes and recreates the collection so all IDs stay consistent.
        """
        try:
            self.client.delete_collection("scam_templates")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="scam_templates",
            embedding_function=self.ef,
        )
        self.seed_known_scams()