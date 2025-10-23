-- Recommended pragmas at connection startup:
-- PRAGMA foreign_keys = ON;
-- PRAGMA journal_mode = WAL;
-- PRAGMA busy_timeout = 5000; -- ms

CREATE TABLE provider (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  // base_url TEXT
);

CREATE TABLE media (
  id INTEGER PRIMARY KEY,
  media_type TEXT NOT NULL,       -- 'movie','episode','tv_show' etc.
  title TEXT,
  imdb_id TEXT UNIQUE,            -- keep but not PK
  // tmdb_id INTEGER,
  original_language TEXT NOT NULL,         -- BCP47/ISO
  duration_seconds INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tmdb_map (
  media_id INTEGER NOT NULL,
  tmdb_id INTEGER NOT NULL,
  PRIMARY KEY (media_id, tmdb_id),
  FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE
);

CREATE TABLE subtitle_file (
  id INTEGER PRIMARY KEY,
  media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
  provider_id INTEGER REFERENCES provider(id),
  provider_sub_id TEXT NOT NULL,
  language TEXT NOT NULL,         -- BCP47/ISO
  format TEXT,                    -- 'srt','vtt','ass'
  storage_url TEXT,               -- path or s3://...
  sha256 TEXT,
  downloaded_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(provider_id, provider_sub_id)
);

-- minimal analysis runs: only speech_seconds metric + params + executed time
CREATE TABLE subtitle_analysis_run (
  id INTEGER PRIMARY KEY,
  subtitle_file_id INTEGER REFERENCES subtitle_file(id) ON DELETE CASCADE,
  parameters TEXT,          -- JSON as TEXT (nullable)
  speech_seconds REAL NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Useful indexes
// CREATE INDEX idx_media_imdb ON media(imdb_id);
// CREATE INDEX idx_sub_sha256 ON subtitle_file(sha256);
// CREATE INDEX idx_sub_language ON subtitle_file(language);
// CREATE INDEX idx_analysis_sub ON analysis_run(subtitle_file_id);
