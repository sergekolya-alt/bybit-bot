from __future__ import annotations

import logging

import requests


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

    def send(self, message: str) -> None:
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Telegram notification failed: %s", exc)
