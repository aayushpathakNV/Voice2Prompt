"""
Global hotkey listener for Voice2Prompt — Windows implementation.

Uses Win32 RegisterHotKey + GetAsyncKeyState via ctypes.
No admin rights required. No external dependencies beyond stdlib.

Works even when the terminal / app window is NOT in focus,
exactly like WhisperFlow's push-to-talk behaviour.

Usage
-----
    from voice2prompt.hotkey import wait_for_hotkey_press, is_hotkey_held

    wait_for_hotkey_press("alt+p")   # blocks until Alt+P is pressed
    while is_hotkey_held("alt+p"):   # returns True while held
        record_audio_chunk()
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import time

if sys.platform != "win32":
    raise ImportError("voice2prompt.hotkey is Windows-only")

user32 = ctypes.WinDLL("user32", use_last_error=True)

# Win32 constants
WM_HOTKEY = 0x0312
PM_REMOVE  = 0x0001

# Modifier flags for RegisterHotKey
_MOD = {
    "alt":   0x0001,
    "ctrl":  0x0002,
    "shift": 0x0004,
    "win":   0x0008,
}

# Virtual-key codes for modifier keys (used with GetAsyncKeyState)
_VK_MOD = {
    "alt":   0xA4,   # VK_LMENU  (left Alt)
    "ctrl":  0xA2,   # VK_LCONTROL
    "shift": 0xA0,   # VK_LSHIFT
    "win":   0x5B,   # VK_LWIN
}


def _parse_hotkey(hotkey: str) -> tuple[int, int]:
    """
    Parse "alt+p" → (mod_flags, vk_code).
    Raises ValueError on unknown tokens.
    """
    parts = [p.strip().lower() for p in hotkey.split("+")]
    main = parts[-1]
    mods = parts[:-1]

    mod_flags = 0
    for m in mods:
        if m not in _MOD:
            raise ValueError(f"Unknown modifier '{m}' in hotkey '{hotkey}'")
        mod_flags |= _MOD[m]

    if len(main) == 1:
        vk = ord(main.upper())
    else:
        # Named keys e.g. "f1", "space"
        named = {
            "space": 0x20, "enter": 0x0D, "tab": 0x09,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
            "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
            "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
        }
        if main not in named:
            raise ValueError(f"Unknown key '{main}' in hotkey '{hotkey}'")
        vk = named[main]

    return mod_flags, vk


def _async_key_down(vk: int) -> bool:
    """Return True if the given virtual-key is currently held down."""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def is_hotkey_held(hotkey: str) -> bool:
    """
    Return True while ALL keys in the hotkey combo are physically held.
    Poll this in a tight loop to detect release.
    """
    parts = [p.strip().lower() for p in hotkey.split("+")]
    main = parts[-1]
    mods = parts[:-1]

    _, vk_main = _parse_hotkey(hotkey)

    if not _async_key_down(vk_main):
        return False
    for m in mods:
        vk_m = _VK_MOD.get(m)
        # also check generic alt (VK_MENU = 0x12) in case right-alt is used
        generic = {"alt": 0x12, "ctrl": 0x11, "shift": 0x10}.get(m)
        held = _async_key_down(vk_m) or (generic and _async_key_down(generic))
        if not held:
            return False
    return True


def wait_for_hotkey_press(hotkey: str = "alt+p", hotkey_id: int = 1) -> None:
    """
    Block until the hotkey combo is pressed once (key-down event).

    Uses RegisterHotKey + PeekMessage so it works globally without admin.
    Automatically unregisters after the first press.
    """
    mod_flags, vk = _parse_hotkey(hotkey)

    if not user32.RegisterHotKey(None, hotkey_id, mod_flags, vk):
        err = ctypes.get_last_error()
        raise OSError(
            f"RegisterHotKey failed (error {err}). "
            "Another app may already own this hotkey — try a different combo."
        )

    try:
        msg = ctypes.wintypes.MSG()
        while True:
            # PeekMessage is non-blocking; sleep to avoid 100% CPU
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == WM_HOTKEY and msg.wParam == hotkey_id:
                    return
            time.sleep(0.005)
    finally:
        user32.UnregisterHotKey(None, hotkey_id)
