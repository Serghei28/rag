"""Grabber: автоматический сбор документов из русской Википедии через MediaWiki API."""

from rag.grabber.grabber import Grabber, GrabResult
from rag.grabber.store import DocumentRecord, Manifest, RawStore

__all__ = ["DocumentRecord", "GrabResult", "Grabber", "Manifest", "RawStore"]
