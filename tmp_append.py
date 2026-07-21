import os

nonscam_pos = r"c:\Users\lanovo\Audio-forensics-Operation-safe-vault\Downloads\Operation-Safe-Vault-main\Operation-Safe-Vault-main\English_NonScam.txt"
scam_pos = r"c:\Users\lanovo\Audio-forensics-Operation-safe-vault\Downloads\Operation-Safe-Vault-main\Operation-Safe-Vault-main\English_Scam.txt"

new_nonscam_lines = [
    "",
    "[Greetings], for your safety, the bank will never ask you to share your OTP or money. Keep your details safe.",
    "",
    "[Greetings], do not disclose your transaction OTP, bank password, or credit card number to anyone. We will never ask for them.",
    "",
    "[Greetings], here is your money. I am returning the cash that I borrowed yesterday for my grocery shopping.",
    "",
    "[Greetings], my salary has been credited today, so I have enough money to pay the electricity bill and transfer rent.",
    "",
    "[Greetings], your refund of fifty dollars has been processed. Please check your bank account or card statement.",
    "",
    "[Greetings], if you receive any suspicious call asking for the OTP of your account, please ignore it or report to police.",
    "",
    "[Greetings], we need to clear the pending invoice of eighty dollars today for the software license key renewal feed.",
    "",
    "[Greetings], I am sending you the payment confirmation receipt for your records. Let me know if you received the money.",
    "",
    "[Greetings], you can transfer the money to your brother via standard bank transfer or UPI app safely.",
    "",
    "[Greetings], your OTP code for login on our official bank application is 456782. Please do not share it with anyone.",
    ""
]

new_scam_lines = [
    "",
    "401.\t[Greetings], please share the OTP you received on your phone to complete your verification immediately otherwise we block.",
    "",
    "402.\t[Greetings], give me the OTP code now, otherwise your ATM card and your bank account will be blocked today.",
    "",
    "403.\t[Greetings], to verify your phone number and release the prize money, you must read out the OTP to me right now.",
    "",
    "404.\t[Greetings], send money immediately to this secure bank account of the CBI department to avoid legal warrant or arrest.",
    "",
    "405.\t[Greetings], confirm your OTP code to prevent cancellation of your loan approval and processing fee waiver.",
    "",
    "406.\t[Greetings], you need to transfer money to this virtual account to verify your transaction history and clear customs.",
    "",
    "407.\t[Greetings], give me the OTP sent to your mobile, this is your bank assistant verifying your security updates.",
    "",
    "408.\t[Greetings], transfer money now or you will be arrested under digital arrest rules by CBI video call link.",
    "",
    "409.\t[Greetings], we detected custom duties on your parcel, pay processing fee or scan this QR code immediately.",
    "",
    "410.\t[Greetings], tell me your banking password and share the OTP to approve your refund of money code.",
    ""
]

with open(nonscam_pos, "a", encoding="utf-8") as f:
    f.write("\n".join(new_nonscam_lines) + "\n")
print("Appended non-scam phrases successfully.")

with open(scam_pos, "a", encoding="utf-8") as f:
    f.write("\n".join(new_scam_lines) + "\n")
print("Appended scam phrases successfully.")
