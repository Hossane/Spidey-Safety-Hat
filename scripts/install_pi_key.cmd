@echo off
rem One-time: copy the laptop's SSH public key to the Pi (asks for the qnxuser password once).
rem Uses the "qnxpi" entry in ~/.ssh/config (aes128-ctr + hmac-sha2-256 to avoid "Corrupted MAC").
ssh qnxpi "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED" < "%USERPROFILE%\.ssh\id_ed25519_qnxpi.pub"
