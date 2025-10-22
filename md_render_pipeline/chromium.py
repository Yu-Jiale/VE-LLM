from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional

from playwright.sync_api import Playwright, sync_playwright


class RenderSession(AbstractContextManager):
    """
    Persistent Chromium session so we avoid launching a browser for every single render.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        chromium_args: Optional[list[str]] = None,
        locale: str = "zh-CN",
        device_scale_factor: float = 1.0,
    ) -> None:
        self._manager: Optional[Playwright] = None
        self._browser = None
        self._headless = headless
        self._chromium_args = chromium_args or ["--no-sandbox"]
        self._locale = locale
        self._device_scale_factor = device_scale_factor
        self._context = None

    def __enter__(self) -> "RenderSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        if self._manager is not None:
            return
        self._manager = sync_playwright().start()
        self._browser = self._manager.chromium.launch(
            headless=self._headless,
            args=self._chromium_args,
        )
        self._context = self._browser.new_context(
            locale=self._locale,
            device_scale_factor=self._device_scale_factor,
        )

    def new_page(self):
        if self._context is None:
            raise RuntimeError("RenderSession has not been started.")
        return self._context.new_page()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._manager is not None:
            self._manager.stop()
            self._manager = None

