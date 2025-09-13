"""SQLite schema and high-level database helpers for the immersion tool (MVP).

Tables:
- content: minimal provider-agnostic metadata for stored subtitles/transcripts
- analysis: computed metrics per content (kept minimal for versioning)

MVP content fields:
- platform, platform_id (nullable), url, title, language, subtitle_path, fetched_at, extra_json
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..config import DB_PATH


def _dict_factory(cursor: sqlite3.Cursor, row: Iterable[Any]):
	"""Return each SQLite row as a dict keyed by column name."""
	return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_connection() -> sqlite3.Connection:
	"""Open (and create if needed) the SQLite DB and enable useful pragmas."""
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(DB_PATH)
	conn.row_factory = _dict_factory
	conn.execute("PRAGMA foreign_keys = ON;")
	return conn


def init_db() -> None:
	"""Create required tables if they do not already exist (MVP schema)."""
	conn = get_connection()
	with conn:
		conn.executescript(
			(
				"""
				CREATE TABLE IF NOT EXISTS content (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					platform TEXT NOT NULL,
					platform_id TEXT,
					url TEXT,
					title TEXT,
					language TEXT NOT NULL,
					subtitle_path TEXT NOT NULL,
					fetched_at TEXT,
					extra_json TEXT,
					UNIQUE(platform, platform_id, language)
				);

				CREATE TABLE IF NOT EXISTS analysis (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					content_id INTEGER NOT NULL,
					metrics_json TEXT NOT NULL,
					created_at TEXT NOT NULL,
					FOREIGN KEY(content_id) REFERENCES content(id) ON DELETE CASCADE
				);
				"""
			)
		)


def upsert_content(meta: Dict[str, Any]) -> int:
	"""Insert or update a content row and return its ``id`` (MVP fields only).

	Rows are identified by (platform, platform_id, language). If a matching row
	exists, it is updated; otherwise a new row is inserted.
	"""
	conn = get_connection()
	with conn:
		cur = conn.execute(
			"""
			SELECT id FROM content WHERE platform = ? AND platform_id IS ? AND language = ?
			""",
			(meta.get("platform"), meta.get("platform_id"), meta.get("language")),
		)
		existing = cur.fetchone()
		encoded_extra = (
			json.dumps(meta.get("extra_json")) if meta.get("extra_json") is not None else None
		)
		if existing:
			content_id = existing["id"]
			fields = [
				"url",
				"title",
				"subtitle_path",
				"fetched_at",
				"extra_json",
			]
			assignments = ", ".join(f"{f} = ?" for f in fields)
			values = [
				meta.get("url"),
				meta.get("title"),
				meta.get("subtitle_path"),
				meta.get("fetched_at"),
				encoded_extra,
			]
			values.append(content_id)
			conn.execute(f"UPDATE content SET {assignments} WHERE id = ?", values)
			return content_id
		else:
			cur = conn.execute(
				"""
				INSERT INTO content (
					platform, platform_id, url, title, language, subtitle_path, fetched_at, extra_json
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				""",
				(
					meta.get("platform"),
					meta.get("platform_id"),
					meta.get("url"),
					meta.get("title"),
					meta.get("language"),
					meta.get("subtitle_path"),
					meta.get("fetched_at"),
					encoded_extra,
				),
			)
			return int(cur.lastrowid)


def get_content_by_id(content_id: int) -> Optional[Dict[str, Any]]:
	"""Return a single content row by numeric ``id`` or ``None`` if not found."""
	conn = get_connection()
	cur = conn.execute("SELECT * FROM content WHERE id = ?", (content_id,))
	return cur.fetchone()


def get_content_by_platform(platform: str, platform_id: Optional[str], language: str) -> Optional[Dict[str, Any]]:
	"""Return a content row by natural key or ``None`` if not found."""
	conn = get_connection()
	cur = conn.execute(
		"SELECT * FROM content WHERE platform = ? AND platform_id IS ? AND language = ?",
		(platform, platform_id, language),
	)
	return cur.fetchone()


def list_content(limit: int = 100) -> List[Dict[str, Any]]:
	"""Return a list of recent content rows (subset of columns) ordered by id desc."""
	conn = get_connection()
	cur = conn.execute(
		"SELECT id, platform, platform_id, language, title, subtitle_path FROM content ORDER BY id DESC LIMIT ?",
		(limit,),
	)
	return list(cur.fetchall())


def insert_analysis(content_id: int, metrics: Dict[str, Any]) -> int:
	"""Insert an analysis record for ``content_id`` and return the new analysis id."""
	conn = get_connection()
	with conn:
		cur = conn.execute(
			"INSERT INTO analysis (content_id, metrics_json, created_at) VALUES (?, ?, ?)",
			(
				content_id,
				json.dumps(metrics, ensure_ascii=False),
				datetime.utcnow().isoformat(timespec="seconds") + "Z",
			),
		)
		return int(cur.lastrowid)
